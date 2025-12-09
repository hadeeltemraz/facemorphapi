from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
from io import BytesIO

app = FastAPI(title="Face Morphing API", version="1.0.0")

# Enable CORS for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def get_face_landmarks(image):
    """Extract facial landmarks from image"""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    h, w = image.shape[:2]
    landmarks = []
    
    # Key facial feature indices
    key_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
        33, 133, 160, 159, 158, 157, 173, 362, 263, 387, 386, 385,
        1, 2, 98, 327, 168, 6, 197, 195, 5,
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146
    ]
    
    # Remove duplicates
    key_indices = list(set(key_indices))
    
    face_landmarks = results.multi_face_landmarks[0]
    for idx in sorted(key_indices):
        if idx < len(face_landmarks.landmark):
            lm = face_landmarks.landmark[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            landmarks.append([x, y])
    
    # Add corner points
    landmarks.extend([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1],
                     [w//2, 0], [w//2, h-1], [0, h//2], [w-1, h//2]])
    
    return np.array(landmarks, dtype=np.float32)

def get_triangulation(points, img_shape):
    """Create Delaunay triangulation"""
    h, w = img_shape[:2]
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    
    for pt in points:
        try:
            subdiv.insert((float(pt[0]), float(pt[1])))
        except:
            pass
    
    triangles = []
    for t in subdiv.getTriangleList():
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        indices = []
        
        for pt in pts:
            min_dist = float('inf')
            min_idx = -1
            for i, p in enumerate(points):
                dist = np.sqrt((p[0] - pt[0])**2 + (p[1] - pt[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            
            if min_dist < 3.0 and min_idx != -1:
                indices.append(min_idx)
        
        if len(indices) == 3 and len(set(indices)) == 3:
            triangles.append(indices)
    
    return triangles

def warp_triangle(img1, img2, img_out, t1, t2, t_out, alpha):
    """Warp and blend triangle"""
    try:
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        r_out = cv2.boundingRect(np.float32([t_out]))
        
        if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0 or r_out[2] <= 0 or r_out[3] <= 0:
            return
        
        t1_rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
        t2_rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
        t_out_rect = [(t_out[i][0] - r_out[0], t_out[i][1] - r_out[1]) for i in range(3)]
        
        mask = np.zeros((r_out[3], r_out[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t_out_rect), (1, 1, 1))
        
        img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        img2_rect = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
        
        if img1_rect.size == 0 or img2_rect.size == 0:
            return
        
        size = (r_out[2], r_out[3])
        warp_mat1 = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t_out_rect))
        warp_mat2 = cv2.getAffineTransform(np.float32(t2_rect), np.float32(t_out_rect))
        
        img1_warped = cv2.warpAffine(img1_rect, warp_mat1, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        img2_warped = cv2.warpAffine(img2_rect, warp_mat2, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        blended = (1.0 - alpha) * img1_warped.astype(float) + alpha * img2_warped.astype(float)
        
        img_out[r_out[1]:r_out[1]+r_out[3], r_out[0]:r_out[0]+r_out[2]] = \
            img_out[r_out[1]:r_out[1]+r_out[3], r_out[0]:r_out[0]+r_out[2]] * (1 - mask) + blended * mask
    except:
        pass

def morph_faces(img1, img2, landmarks1, landmarks2, alpha):
    """Main morphing function"""
    points = (1 - alpha) * landmarks1 + alpha * landmarks2
    img_morph = np.zeros(img1.shape, dtype=img1.dtype)
    triangles = get_triangulation(points, img1.shape)
    
    for tri_indices in triangles:
        t1 = [landmarks1[i] for i in tri_indices]
        t2 = [landmarks2[i] for i in tri_indices]
        t_out = [points[i] for i in tri_indices]
        warp_triangle(img1, img2, img_morph, t1, t2, t_out, alpha)
    
    return img_morph

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Face Morphing API is running!",
        "version": "1.0.0"
    }

@app.post("/morph")
async def morph_images(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    alpha: float = 0.5
):
    """
    Morph two face images together
    
    Parameters:
    - image1: First face image
    - image2: Second face image  
    - alpha: Morph amount (0.0 = 100% image1, 1.0 = 100% image2)
    
    Returns:
    - Morphed image as JPEG
    """
    try:
        # Validate alpha
        if not 0.0 <= alpha <= 1.0:
            raise HTTPException(status_code=400, detail="Alpha must be between 0.0 and 1.0")
        
        # Read images
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()
        
        # Convert to numpy arrays
        img1_array = np.frombuffer(img1_bytes, np.uint8)
        img2_array = np.frombuffer(img2_bytes, np.uint8)
        
        img1 = cv2.imdecode(img1_array, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(img2_array, cv2.IMREAD_COLOR)
        
        if img1 is None or img2 is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Resize to same size
        target_size = (800, 800)
        img1 = cv2.resize(img1, target_size)
        img2 = cv2.resize(img2, target_size)
        
        # Detect landmarks
        landmarks1 = get_face_landmarks(img1)
        landmarks2 = get_face_landmarks(img2)
        
        if landmarks1 is None:
            raise HTTPException(status_code=400, detail="No face detected in image 1")
        
        if landmarks2 is None:
            raise HTTPException(status_code=400, detail="No face detected in image 2")
        
        # Perform morphing
        result = morph_faces(img1, img2, landmarks1, landmarks2, alpha)
        
        # Encode result as JPEG
        success, encoded_image = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode result image")
        
        # Return as image response
        return Response(content=encoded_image.tobytes(), media_type="image/jpeg")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Morphing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    return {"status": "healthy", "service": "face-morph-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)