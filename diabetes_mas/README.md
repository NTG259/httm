# diabetes_mas — Hệ đa tác tử KG + RAG cho tiểu đường

Tài liệu thiết kế: [`../docs/TAI_LIEU_PHAT_TRIEN.md`](../docs/TAI_LIEU_PHAT_TRIEN.md).

## Cài đặt

```bash
cd diabetes_mas
python -m pip install -e ".[kb,dev]"        # thêm [embed] khi cần tạo embedding (tải ~2,3 GB model)
cp .env.example .env                        # rồi đặt NEO4J_PASSWORD, SEARXNG_SECRET
```

## Giai đoạn 0 — Dữ liệu & KB

```bash
python -m diabetes_mas.ingest.fetch_vinmec      # ~15 phút (giới hạn 1 request/2 giây); chạy lại được
python -m diabetes_mas.ingest.fetch_static      # PDF Bộ Y tế, WHO, Hetionet (configs/sources.yaml)
python -m diabetes_mas.ingest.clean             # → data/interim/docs.jsonl, data/processed/sources.jsonl
python -m diabetes_mas.ingest.chunk             # → data/processed/chunks.jsonl
python -m diabetes_mas.kb.seed                  # kiểm chứng trích dẫn của seed KB (exit 1 nếu sai)
python -m diabetes_mas.kb.build_cypher          # → kb/kb_diabetes.cypher, data/processed/kb_stats.json
python -m diabetes_mas.simulator.cases          # → data/cases/{train,val,test}.jsonl

docker compose up -d neo4j                      # cần Docker Desktop đang chạy
python -m diabetes_mas.kb.neo4j_io load         # nạp schema + KB, in báo cáo chất lượng
python -m diabetes_mas.rag.embed_chunks         # ghi embedding bge-m3 cho Chunk
```

Nạp bằng `cypher-shell` (cách trình bày khi demo). Thư mục `kb/` đã mount vào container tại `/import`:

```bash
docker compose exec neo4j sh -c 'cypher-shell -u neo4j -p "$KB_PASS" -f /import/schema.cypher'
docker compose exec neo4j sh -c 'cypher-shell -u neo4j -p "$KB_PASS" -f /import/kb_diabetes.cypher'
```

> Không dùng `cat ... | cypher-shell` từ Windows PowerShell 5.1: PowerShell mã hóa dữ liệu pipe sang ASCII → toàn bộ tiếng Việt thành `?` mà lệnh vẫn báo thành công.

## Kiểm thử

```bash
python -m pytest -q tests/unit
```

## Sửa KB

Chỉ sửa `kb/seed/*.csv`. Mỗi quan hệ cần `source_id` và `evidence` là câu trích **nguyên văn** từ nguồn đó. Sau khi sửa, chạy lại `kb.seed`, `kb.build_cypher` và `simulator.cases`. Không sửa tay `kb/kb_diabetes.cypher`.
