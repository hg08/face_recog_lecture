import face_recognition

# step1
"""
image = face_recognition.load_image_file("WP_004050.jpg")
face_locations = face_recognition.face_locations(image)
print(face_locations)
"""
# step2
"""
image = face_recognition.load_image_file("WP_004050.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)
print(face_landmarks_list)
"""

# step3
known_image = face_recognition.load_image_file("01.jpg")
#unknown_image = face_recognition.load_image_file("20K-2011.jpg")
#unknown_image = face_recognition.load_image_file("DSC_8384.JPG")
#unknown_image = face_recognition.load_image_file("0403_3b.jpg")
#unknown_image = face_recognition.load_image_file("WP2498.jpg")
#unknown_image = face_recognition.load_image_file("IMG1324.JPG")
unknown_image = face_recognition.load_image_file("03.jpg")

huang_encoding = face_recognition.face_encodings(known_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
            huang_encoding,
                ]
known_face_names = [
            "Huang Gang"
                ]


unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([huang_encoding], unknown_encoding)
print(results)
