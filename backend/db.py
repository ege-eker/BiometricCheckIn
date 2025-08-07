from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", 5432)
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "face_recognition")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

DB_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

engine = create_engine(DB_URL)


# Functions for database operations

def find_most_similar_face(embedding, top_k=3):
    with engine.connect() as conn:
        res = conn.execute(
            text("""
                WITH person_matches AS (
                    SELECT
                        p.id, p.name, p.surname, p.age, p.nationality, p.flight_no, p.passport_no,
                        1 - (fe.embedding <=> (:embedding)::vector) AS similarity,
                        ROW_NUMBER() OVER (PARTITION BY p.id ORDER BY fe.embedding <=> (:embedding)::vector ASC) as rank
                    FROM people p
                    JOIN face_embeddings fe ON p.id = fe.person_id
                )
                SELECT id, name, surname, age, nationality, flight_no, passport_no, similarity
                FROM person_matches
                WHERE rank = 1  -- best match for each person
                ORDER BY similarity DESC
                LIMIT :top_k
            """),
            dict(embedding=embedding.tolist(), top_k=top_k)
        )
        rows = res.fetchall()

        if not rows:
            return None

        # Get the best match
        best_match = rows[0]
        similarity = float(best_match[7])

        # Confidence score boost, %70 threshold for each embedding match
        person_id = best_match[0]
        confidence_query = conn.execute(
            text("""
                SELECT COUNT(*) 
                FROM face_embeddings 
                WHERE person_id = :person_id 
                AND (1 - (embedding <=> (:embedding)::vector)) > 0.70
            """),
            dict(person_id=person_id, embedding=embedding.tolist())
        )
        good_match_count = confidence_query.scalar()
        print(f"Good match count for person {person_id}: {good_match_count}")
        confidence_boost = min(0.1, (good_match_count * 0.02))  # Max 0.1 boost, can be adjusted
        print(f"Confidence boost: {confidence_boost}")
        adjusted_similarity = min(1.0, similarity + confidence_boost)
        print(f"original similarity: {similarity}, adjusted similarity: {adjusted_similarity}")

        # Return the best match details
        return {
            "name": best_match[1],
            "surname": best_match[2],
            "age": best_match[3],
            "nationality": best_match[4],
            "flight_no": best_match[5],
            "passport_no": best_match[6],
            "similarity": adjusted_similarity
        }


def db_get_person_by_passport(passport_no):
    # deprecated: this function is no longer used in the codebase but would be useful for debugging or future reference
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT id FROM people WHERE passport_no = :pass"),
            {"pass": passport_no}
        )
        return res.fetchone()


def db_insert_person(info):
    # turn embedding into list ndarray
    eb = info["embedding"]
    if hasattr(eb, "tolist"):
        info["embedding"] = eb.tolist()

    with engine.begin() as conn:
        # 1. Add the person first
        result = conn.execute(
            text("""
                INSERT INTO people
                (name, surname, age, nationality, flight_no, passport_no)
                VALUES (:name, :surname, :age, :nationality, :flight_no, :passport_no)
                RETURNING id
            """),
            {k: v for k, v in info.items() if k != "embedding"}
        )

        # Print the person ID of the newly created person
        person_id = result.scalar()
        print(f"DB: Inserted person ID: {person_id}")

        # 2. Then add the embedding
        conn.execute(
            text("""
                INSERT INTO face_embeddings
                (person_id, embedding)
                VALUES (:person_id, :embedding)
            """),
            {"person_id": person_id, "embedding": info["embedding"]}
        )

        # Return the person ID
        return person_id


def db_check_person_exists(person_id):
    """Checks if a person with the given ID exists in the database"""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT id FROM people WHERE id = :person_id"),
            {"person_id": person_id}
        )
        return result.fetchone() is not None


def db_add_embedding(person_id, embedding):
    """Kişiye yeni bir embedding ekler"""
    # Turn embedding into list, numpy
    if hasattr(embedding, "tolist"):
        embedding_list = embedding.tolist()
    else:
        embedding_list = embedding

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO face_embeddings
                (person_id, embedding)
                VALUES (:person_id, :embedding)
            """),
            {
                "person_id": person_id,
                "embedding": embedding_list
            }
        )


def db_delete_person(person_id):
    """Deletes a person and all their embeddings from the database"""
    if person_id is None:
        return False

    try:
        with engine.begin() as conn:
            # First delete all embeddings for the person
            conn.execute(
                text("DELETE FROM face_embeddings WHERE person_id = :person_id"),
                {"person_id": person_id}
            )

            # Then delete the person record
            conn.execute(
                text("DELETE FROM people WHERE id = :person_id"),
                {"person_id": person_id}
            )

        return True
    except Exception as e:
        print(f"Person deleteion error: {str(e)}")
        return False
