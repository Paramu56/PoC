Data directory for ingestion
============================

1) PDF — Karnataka schemes (recommended file name)
   Place your main PDF here as:
      Karnataka Schemes.pdf

   The ingest script looks here first, then falls back to a single PDF in this folder,
   then legacy paths under Downloads.

2) Cypher — exclusions and relationships (optional but recommended)
   Place your graph rules file here as:
      schemes.cypher

   On ingest, statements are compiled to:
      graph_compiled.json

   At query time (app.py / orchestrated_rag_schemes.py), statements that mention
   retrieved scheme names (plus global exclusion-style lines) are appended for Gemini.

3) Re-ingest after you change the PDF or schemes.cypher
   Run:
      python ingest_karnataka_schemes.py

   To DROP all old vectors and use ONLY the new PDF (recommended when replacing the file):
      python ingest_karnataka_schemes.py --fresh

   Use --pdf "path\to\other.pdf" if your file is not in this folder.
   Use --no-cypher to skip compiling the Cypher file.
