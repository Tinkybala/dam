from demo.i18n import TEXT, text


def test_translation_tables_have_matching_keys():
    assert set(TEXT["zh"]) == set(TEXT["en"])


def test_user_labels_are_localized():
    assert text("zh", "anonymous_user", number=1) == "匿名用户 01"
    assert text("en", "anonymous_user", number=20) == "Anonymous User 20"


def test_language_switch_labels_point_to_other_language():
    assert text("zh", "switch_language") == "English"
    assert text("en", "switch_language") == "中文"
