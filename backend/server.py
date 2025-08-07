import grpc
from concurrent import futures
import base64
import cv2
import numpy as np
import time
import os

import facerecognizer_pb2 as pb2
import facerecognizer_pb2_grpc as pb2_grpc
from db import db_insert_person, db_get_person_by_passport, find_most_similar_face, db_check_person_exists, \
    db_add_embedding, db_delete_person

from dotenv import load_dotenv

load_dotenv()
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", 0.80))  # Minimum similarity threshold for face recognition, default is 0.80. Can be adjusted in .env file.

from embedding_model import EmbeddingModel

embedding_model = EmbeddingModel()


class FaceRecognizerService(pb2_grpc.FaceRecognizerServicer):
    def Recognize(self, request, context):
        try:
            image_data = base64.b64decode(request.image_base64)

            with open("recived_image.jpg", "wb") as f:  # Debugging
                f.write(image_data)

            np_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if image is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('Image data is not a valid image (decode error).')
                return pb2.FaceResponse()
            embedding = embedding_model.get_embedding(image)
            if embedding is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details('No face detected in the image')
                return pb2.FaceResponse()

            result = find_most_similar_face(embedding)
            if not result:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details('No match found in database')
                return pb2.FaceResponse()

            if result["similarity"] < MIN_SIMILARITY:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                print("similarity:", result["similarity"], " similiar person:", result["name"], result["surname"])
                context.set_details('Below minimum similarity threshold.')
                return pb2.FaceResponse()

            # Found a match, return the response, flight_no can be empty
            return pb2.FaceResponse(
                name=result["name"],
                surname=result["surname"],
                age=result["age"],
                nationality=result["nationality"],
                flight_no=result["flight_no"] or "",
                passport_no=result["passport_no"],
                similarity=result["similarity"],
            )
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details('An error occured.')
            return pb2.FaceResponse()

    def RegisterPerson(self, request, context):
        try:
            # 1. Decoding and embedding
            image_data = base64.b64decode(request.image_base64)
            np_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if image is None:
                return pb2.RegisterPersonResponse(
                    success=False,
                    message="image is not a valid image (decode error)."
                )
            embedding = embedding_model.get_embedding(image)
            if embedding is None:
                return pb2.RegisterPersonResponse(
                    success=False,
                    message="could not extract embedding (no face detected)."
                )

            # 2. Duplicate passport number check(can be UNIQUE constraint in DB)
            # existing = db_get_person_by_passport(request.passport_no)
            # if existing:
            #   return pb2.RegisterPersonResponse(
            #        success=False,
            #        message="Bu pasaport numarası ile kayıt var!"
            #    )

            # 3. DB INSERT
            person_id = db_insert_person({
                'name': request.name,
                'surname': request.surname,
                'age': request.age,
                'nationality': request.nationality,
                'flight_no': request.flight_no,
                'passport_no': request.passport_no,
                'embedding': embedding
            })

            # Debugging: print the person_id
            print(f"Server: Returning person id from db_insert_person: {person_id}")

            response = pb2.RegisterPersonResponse(
                success=True,
                message="Person successfully registered.",
                person_id=person_id
            )

            # Debugging: print the person_id from the response
            print(f"Server: Person id in response: {response.person_id}")

            return response

        except Exception as e:
            print(f"[ERROR] {str(e)}") # Log the error on the server side
            return pb2.RegisterPersonResponse(
                success=False,
                message="An error occured.",
                person_id=0
            )

    def AddEmbedding(self, request, context):
        """
        Service to add a new embedding for an existing person.
        """
        try:
            # 1 Decode the image from base64
            image_data = base64.b64decode(request.image_base64)
            np_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if image is None:
                return pb2.AddEmbeddingResponse(
                    success=False,
                    message="Decode error."
                )

            # 2 Getting the embedding from the image
            embedding = embedding_model.get_embedding(image)
            if embedding is None:
                return pb2.AddEmbeddingResponse(
                    success=False,
                    message="Can't detect face on image."
                )

            # 3. Check if the person exists in the database
            person_id = request.person_id
            if not db_check_person_exists(person_id):
                return pb2.AddEmbeddingResponse(
                    success=False,
                    message=f"Didn't find a person with ID:{person_id}."
                )

            # 4. Add the new embedding to the database
            db_add_embedding(person_id, embedding)

            return pb2.AddEmbeddingResponse(
                success=True,
                message=f"Embedding successfully added for person ID: {person_id}.",
            )


        except Exception as e:

            print(f"[ERROR] {str(e)}")  # Log the error on the server side

            return pb2.RegisterPersonResponse(
                success=False,
                message="An error occurred while adding embedding.",
                person_id=0
            )

    def RegisterCompletePerson(self, request, context):
        try:
            images = request.images

            if len(images) < 5:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Not enough photo count. Min: 5, Got: {len(images)}.")
                return pb2.RegisterCompletePersonResponse(
                    success=False,
                    message=f"Not enough photo count. Min: 5, Got: {len(images)}.",
                    person_id=0
                )

            # Get first images embedding
            first_image = images[0]
            image_data = base64.b64decode(first_image)
            np_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if image is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("First image is not a valid image (decode error).")
                return pb2.RegisterCompletePersonResponse(
                    success=False,
                    message="First image is not a valid image (decode error).",
                    person_id=0
                )

            embedding = embedding_model.get_embedding(image)
            if embedding is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("Can't detect face on the first image.")
                return pb2.RegisterCompletePersonResponse(
                    success=False,
                    message="Can't detect face on the first image.",
                    person_id=0
                )

            try:
                # Initilaze DB operation
                person_id = db_insert_person({
                    'name': request.name,
                    'surname': request.surname,
                    'age': request.age,
                    'nationality': request.nationality,
                    'flight_no': request.flight_no,
                    'passport_no': request.passport_no,
                    'embedding': embedding
                })

                # Other images processing
                for i, image_base64 in enumerate(images[1:], 1):
                    try:
                        image_data = base64.b64decode(image_base64)
                        np_array = np.frombuffer(image_data, np.uint8)
                        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                        if image is not None:
                            embedding = embedding_model.get_embedding(image)
                            if embedding is not None:
                                db_add_embedding(person_id, embedding)
                            else:
                                print(f"Can't detect face in {i + 1}. embedding.")
                    except Exception as e:
                        print(f"Error in {i + 1}. embedding: {str(e)}")

                return pb2.RegisterCompletePersonResponse(
                    success=True,
                    message="Person successfully registered with multiple images.",
                    person_id=person_id
                )

            except Exception as e:
                # When an error occurs, delete the person from the database
                db_delete_person(person_id if 'person_id' in locals() else None)

                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Error in register process. Reverting changes in database: {str(e)}")
                return pb2.RegisterCompletePersonResponse(
                    success=False,
                    message=f"Error in register process. Reverting changes in database: {str(e)}",
                    person_id=0
                )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Unexpected error: {str(e)}")
            return pb2.RegisterCompletePersonResponse(
                success=False,
                message=f"Unexpected error: {str(e)}",
                person_id=0
            )


def serve():
    # Create a gRPC server and add the FaceRecognizerService to it
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10)) # Adjust max_workers as needed
    pb2_grpc.add_FaceRecognizerServicer_to_server(FaceRecognizerService(), server)

    server.add_insecure_port('[::]:50051') # Listen on all interfaces on port 50051, no TLS encryption
    server.start()

    print("Server started on port 50051")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
