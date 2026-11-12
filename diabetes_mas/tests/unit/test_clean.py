from diabetes_mas.ingest.clean import find_reviewer


def test_find_reviewer_detects_doctor_line():
    texts = ["Bài viết được tư vấn chuyên môn bởi Thạc sĩ, Bác sĩ Vũ Thị Duyên - Khoa Khám bệnh.\nNội dung..."]
    assert find_reviewer(texts) == "Thạc sĩ, Bác sĩ Vũ Thị Duyên - Khoa Khám bệnh"


def test_find_reviewer_none_when_absent():
    assert find_reviewer(["Đái tháo đường là bệnh mạn tính."]) is None
