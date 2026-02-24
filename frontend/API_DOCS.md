# LPR Vision API Documentation

This document describes the REST APIs exposed by the License Plate Recognition backend service. The frontend application relies on these endpoints to function properly.

## Base URL
`http://<SERVER_IP>:<PORT>/api/v1`

---

## 1. Authentication

### 1.1 User Login
Authenticates a user and returns a JWT token.

**Endpoint:** `/auth/login`
**Method:** `POST`
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "username": "string",
  "password": "string(hashed)",
  "rememberMe": "boolean (optional)"
}
```

**Successful Response:** `200 OK`
```json
{
  "code": 200,
  "message": "Success",
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5c...",
    "user": {
      "id": 1,
      "username": "john_doe",
      "email": "name@company.com"
    }
  }
}
```

### 1.2 User Registration
Registers a new user account.

**Endpoint:** `/auth/register`
**Method:** `POST`
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "username": "string",
  "email": "string",
  "password": "string(hashed)"
}
```

**Successful Response:** `201 Created`
```json
{
  "code": 201,
  "message": "User registered successfully",
  "data": null
}
```

---

## 2. Analysis

### 2.1 Upload & Analyze Image
Uploads an image containing a license plate and runs the selected recognition model.

**Endpoint:** `/analyze/upload`
**Method:** `POST`
**Content-Type:** `multipart/form-data`

**Request Parameters (Form Data):**
- `file`: (File) The image file (JPG, PNG). Max size: 10MB.
- `modelType`: (String) The algorithm model to specify. Valid options: `yolov8_fast`, `yolov8_acc`, `resnet50`.

**Successful Response:** `200 OK`
```json
{
  "code": 200,
  "message": "Analysis Complete",
  "data": {
    "recordId": 1024,
    "plateNumber": "沪A·88888",
    "confidence": 0.9856,
    "processingTimeMs": 124,
    "boundingBox": {
      "x": 120,
      "y": 300,
      "width": 200,
      "height": 60
    },
    "thumbnailUrl": "/static/images/thumbs/1024.jpg",
    "resultImageUrl": "/static/images/results/1024.jpg" 
  }
}
```

---

## 3. History Records

### 3.1 Get History List
Retrieves a paginated list of past recognition records for the authenticated user.

**Endpoint:** `/history/list`
**Method:** `GET`
**Headers:** 
- `Authorization: Bearer <token>`

**Query Parameters:**
- `page`: (Integer) Current page number. Default `1`.
- `size`: (Integer) Number of records per page. Default `10`.
- `keyword`: (String, optional) Search query for plate number.
- `startDate`: (String, optional) Keep records after this date (ISO 8601).
- `endDate`: (String, optional) Keep records before this date (ISO 8601).

**Successful Response:** `200 OK`
```json
{
  "code": 200,
  "message": "Success",
  "data": {
    "total": 50,
    "current": 1,
    "size": 10,
    "records": [
      {
        "id": 1,
        "createdAt": "2026-02-24T14:30:22Z",
        "plateNumber": "沪A·88888",
        "modelType": "yolov8_fast",
        "confidence": 0.9856,
        "processingTimeMs": 124,
        "thumbnailUrl": "/static/images/thumbs/1.jpg"
      }
    ]
  }
}
```

### 3.2 Delete History Record
Deletes a specific history record.

**Endpoint:** `/history/{id}`
**Method:** `DELETE`
**Headers:** 
- `Authorization: Bearer <token>`

**Successful Response:** `200 OK`
```json
{
  "code": 200,
  "message": "Record deleted successfully",
  "data": null
}
```

## Error Handling standard
All error responses maintain a consistent format:
```json
{
  "code": 401, // e.g. 401 Unauthorized, 400 Bad Request
  "message": "Detailed error description",
  "data": null
}
```
