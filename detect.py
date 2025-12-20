# Classes
NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Configure which classes to detect by name here:
# Set to None for all classes, or a list of class names (e.g., ["person", "car"])
CLASSES_NAMES = list(NAMES.values()) # Example: ["person"], or None for all classes

def names_to_indices(names_list):
    if names_list is None:
        return None
    name_to_id = {v: k for k, v in NAMES.items()}
    return [name_to_id[name] for name in names_list if name in name_to_id]

CLASSES = names_to_indices(CLASSES_NAMES)
