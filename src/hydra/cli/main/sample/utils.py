from os import listdir

def get_current_sample_path(model_samples_path):
    existing_samples = sorted(map(int, filter(lambda x: x.isdigit(), listdir(model_samples_path))))
    for i, s in enumerate(existing_samples):
        if i != s:
            return str(i)
    return str(len(existing_samples))
