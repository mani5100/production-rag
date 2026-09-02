FILE_CONTEXT_MAP: dict[str, str] = {
    "panel_hospital_list_aicl-updated_115838404328.pdf": (
        "This document lists Nextbridge (NXB) employees' panel/reimbursable "
        "hospitals under Adamjee Insurance Company Limited (AICL) group "
        "medical insurance. Employees can avail cashless/credit treatment "
        "at these hospitals under their NXB medical insurance."
    ),
    "aicl_discount_center_list-updated_209015612052.pdf": (
        "This document lists discount rates NXB employees receive at "
        "diagnostic centres and labs (for tests such as CT scans, MRIs, "
        "lab work) under Adamjee Insurance (AICL), part of Nextbridge's "
        "employee medical insurance benefits."
    ),
    "non_reimburseable_hospital_list_aicl-updated_119123480665.pdf": (
        "This document lists hospitals that are NOT reimbursable for "
        "NXB employees under Adamjee Insurance (AICL) group medical "
        "insurance - i.e. employees should avoid these hospitals for "
        "insurance claims, as treatment there will not be reimbursed."
    ),
}


def get_doc_context(file_name: str) -> str:
    """
    Returns the context statement for a file, matched by exact
    filename or substring, or an empty string if none is configured.
    """
    for key, context in FILE_CONTEXT_MAP.items():
        if key in file_name:
            return context
    return ""