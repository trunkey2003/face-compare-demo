from apiflask.fields import String
from apiflask import APIFlask, Schema
import face_recognition
from flask import jsonify
import tempfile
import os
import base64
from PIL import Image
from io import BytesIO


app = APIFlask(__name__)


class FaceCompareInput(Schema):
    image1_base64 = String()
    image2_base64 = String()


@app.post('/face-compare')
@app.input(FaceCompareInput(partial=True))
def face_compare(req):
    try:
        image1_base64 = req['image1_base64']
        image2_base64 = req['image2_base64']
        # Decode base64 images
        image1_data = base64.b64decode(image1_base64)
        image2_data = base64.b64decode(image2_base64)

        # Create temporary files for the images
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file1, \
            tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file2:
            # Write the image data to the temporary files
            temp_file1.write(image1_data)
            temp_file1.flush()
            temp_file2.write(image2_data)
            temp_file2.flush()

            # Load the images using face_recognition library
            image1_np = face_recognition.load_image_file(temp_file1.name)
            image2_np = face_recognition.load_image_file(temp_file2.name)

            # Encode face features
            image1_face_encodings = face_recognition.face_encodings(image1_np)
            image2_face_encodings = face_recognition.face_encodings(image2_np)
            
            # Compare face encodings
            if len(image1_face_encodings) > 0 and len(image2_face_encodings) > 0:
                # Take the first face encoding (assuming only one face per image)
                image1_face_encoding = image1_face_encodings[0]
                image2_face_encoding = image2_face_encodings[0]
                # Compare the face encodings
                results = face_recognition.compare_faces([image1_face_encoding], image2_face_encoding)
                if results[0]:
                    result = jsonify({"status": "success"})
                else:
                    result = jsonify({"status": "failure", "message": "Faces do not match."})
            else:
                result = jsonify({"status": "failure", "message": "No faces found in one or both images."})
        return result
    except Exception as e:
        return jsonify({"status": "failure", "message": str(e)})
    finally:
        os.remove(temp_file1.name)
        os.remove(temp_file2.name)
