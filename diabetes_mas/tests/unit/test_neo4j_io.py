from diabetes_mas.kb.neo4j_io import split_statements


def test_split_keeps_semicolon_inside_string():
    text = "// chú thích\nMERGE (n:A {t: 'a; b'});\nMATCH (n) RETURN n;\n"
    assert split_statements(text) == ["MERGE (n:A {t: 'a; b'})", "MATCH (n) RETURN n"]
