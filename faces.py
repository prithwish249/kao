import face_recognition

train_file = input("Enter the training file name: ")
image = face_recognition.load_image_file(train_file)
grover_face = face_recognition.face_encodings(image)[0]

test_file = input("Enter the test file: ")
ukn = face_recognition.load_image_file(test_file)
ukn_enc = face_recognition.face_encodings(ukn)[0]

face_locations = face_recognition.face_locations(image)
result = face_recognition.compare_faces([grover_face], ukn_enc)

if result[0]:
	print(result)
	print("This is Grover")
else:
	print("Unknown face")

