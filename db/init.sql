-- Vector database schema for storing people and their face embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Table that holds people information
CREATE TABLE people (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    surname TEXT NOT NULL,
    age INTEGER NOT NULL,
    nationality TEXT NOT NULL,
    flight_no TEXT,   -- Optional flight number
    passport_no TEXT NOT NULL UNIQUE
);

-- Face embeddings table to store facial features with multiple embeddings per person
CREATE TABLE face_embeddings (
    id SERIAL PRIMARY KEY,
    person_id INTEGER REFERENCES people(id) ON DELETE CASCADE,
    embedding VECTOR(512) NOT NULL,
    capture_condition TEXT, -- Light, angle, etc. Currently not in use
    capture_date TIMESTAMP DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX ON face_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX ON face_embeddings (person_id);