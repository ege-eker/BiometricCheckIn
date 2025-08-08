import grpc
import facerecognizer_pb2 as pb2
import facerecognizer_pb2_grpc as pb2_grpc


def send_face(base64_img: str):
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb2_grpc.FaceRecognizerStub(channel)
    request = pb2.FaceRequest(image_base64=base64_img)
    try:
        response = stub.Recognize(request)
        print(
            f"Matching person: {response.name} {response.surname}\n(similarity: {response.similarity})\npasaport no: {response.passport_no}\nage: {response.age}\nflight no: {response.flight_no}")

        # Return the response as a dictionary
        return {
            "name": response.name,
            "surname": response.surname,
            "age": response.age,
            "nationality": response.nationality,
            "passport_no": response.passport_no,
            "flight_no": response.flight_no,
            "similarity": response.similarity
        }
    except grpc.RpcError as rpc_error:
        print(f"GRPC error: {rpc_error.details()}")
        return None


def register_new_person(base64_img, name, surname, age, nationality, flight_no, passport_no):
    """first registers a person, then adds the first embedding"""
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb2_grpc.FaceRecognizerStub(channel)
    request = pb2.RegisterPersonRequest(
        name=name,
        surname=surname,
        age=age,
        nationality=nationality,
        flight_no=flight_no,
        passport_no=passport_no,
        image_base64=base64_img
    )
    response = stub.RegisterPerson(request)
    print("Registered Sucesfully:" if response.success else "Register failed:", response.message)
    print(f"Sucessfull first register, person_id: {response.person_id}")  # debug
    return response


def add_embedding_to_person_by_id(base64_img, person_id):
    """Adds an embedding to a person by their ID."""
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb2_grpc.FaceRecognizerStub(channel)
    request = pb2.AddEmbeddingRequest(
        person_id=person_id,
        image_base64=base64_img
    )
    response = stub.AddEmbedding(request)
    return response


def register_person_with_embeddings(name, surname, age, nationality, flight_no, passport_no, images):
    """
    Registers a person with multiple images and returns the response.
    """
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb2_grpc.FaceRecognizerStub(channel)

    try:
        # Create grpc request
        request = pb2.RegisterCompletePersonRequest(
            name=name,
            surname=surname,
            age=int(age),
            nationality=nationality,
            flight_no=flight_no,
            passport_no=passport_no,
            images=images
        )

        # Send the request to the server
        response = stub.RegisterCompletePerson(request)

        return {
            "success": response.success,
            "message": response.message,
            "person_id": response.person_id if response.success else None
        }
    except grpc.RpcError as e:
        print(f"gRPC error: {e.details()}")
        return {
            "success": False,
            "message": f"Connection error: {e.details()}"
        }
