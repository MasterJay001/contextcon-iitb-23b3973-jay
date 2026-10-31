import pytest

from crust.titles import expand_titles, family_distance, normalize_title, title_similarity


@pytest.mark.parametrize("raw,family,seniority", [
    ("Senior Software Engineer II - Payments", "swe_general", 3),
    ("SDE-3", "swe_general", 3),
    ("SDE 1", "swe_general", 1),
    ("Sr. Backend Engineer", "swe_backend", 3),
    ("Staff Backend Engineer", "swe_backend", 4),
    ("Backend Developer", "swe_backend", 2),
    ("Growth Hacker", "growth_marketing", 2),
    ("Engineering Manager", "eng_manager", 4),
    ("Associate Product Manager", "product_manager", 1),
    ("Product Manager", "product_manager", 2),
    ("Senior Product Manager", "product_manager", 3),
    ("Director of Engineering", "eng_manager", 6),
    ("VP of Engineering", "cto", 7),
    ("Founding Engineer", "swe_general", 3),
    ("Rust Engineer (Embedded)", "embedded_engineer", 2),
    ("Data Scientist II", "data_scientist", 2),
    ("Software Engineer Intern", "swe_general", 0),
    ("Head of Growth", "growth_marketing", 6),
    ("Full-Stack Developer", "swe_fullstack", 2),
    ("Back-end engineer", "swe_backend", 2),
])
def test_normalisation(raw, family, seniority):
    n = normalize_title(raw)
    assert n.family == family, n
    assert n.seniority == seniority, n


def test_unknown_and_empty():
    assert normalize_title(None).family == "unknown"
    assert normalize_title("   ").family == "unknown"
    assert normalize_title("zzzz qqqq").family == "unknown"


def test_api_seniority_hint_only_when_title_silent():
    assert normalize_title("Software Engineer", "Senior").seniority == 3
    assert normalize_title("Junior Software Engineer", "Director").seniority == 1  # explicit title wins


def test_distance_and_similarity_ordering():
    be, fe, sw = normalize_title("Backend Engineer"), normalize_title("Frontend Engineer"), normalize_title("Software Engineer")
    pm = normalize_title("Product Manager")
    assert family_distance("swe_backend", "swe_backend") == 0
    assert family_distance("swe_backend", "swe_general") == 1
    assert title_similarity(be, be) > title_similarity(be, sw) > title_similarity(be, pm)
    assert title_similarity(be, fe) > title_similarity(be, pm)


def test_expand_titles_adds_adjacent_families():
    out = expand_titles(["backend engineer"])
    assert "backend engineer" in out and "software engineer" in out and len(out) >= 8
