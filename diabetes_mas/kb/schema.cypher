// Schema KB tiểu đường — chạy TRƯỚC kb_diabetes.cypher. Xem docs/00_KB_GRAPH.md
CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (n:Disease) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT symptom_id IF NOT EXISTS FOR (n:Symptom) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT labtest_id IF NOT EXISTS FOR (n:LabTest) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT crit_id IF NOT EXISTS FOR (n:DiagnosticCriterion) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT risk_id IF NOT EXISTS FOR (n:RiskFactor) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (n:Drug) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (n:Gene) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT anat_id IF NOT EXISTS FOR (n:Anatomy) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT source_id IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE;

// Liên kết thực thể (T-02); aliases_text gồm bí danh có dấu + bỏ dấu
CREATE FULLTEXT INDEX entity_names IF NOT EXISTS
  FOR (n:Disease|Symptom|Drug|RiskFactor|LabTest) ON EACH [n.name_vi, n.name_en, n.name, n.aliases_text];

// Tìm bài trong kho (T-12, scope = local)
CREATE FULLTEXT INDEX source_text IF NOT EXISTS
  FOR (n:Source|Chunk) ON EACH [n.title, n.text];

// Vector search (T-04); embedding ghi bởi `python -m diabetes_mas.rag.embed_chunks`
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
  FOR (c:Chunk) ON (c.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};
