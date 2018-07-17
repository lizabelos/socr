import numpy as np

from socr.utils.image.connected_components import connected_components


class BSplinesDecoder:

    def __init__(self, height_factor):
        self.height_factor = height_factor

    def decode(self, image, predicted, with_images=True, degree=3):
        components = connected_components(predicted[0])
        num_components = np.max(components)
        results = []

        for i in range(1, num_components + 1):
            result = self.process_components(image, predicted, components, i, with_image=with_images, degree=degree, line_height=64, baseline_resolution=16)
            if result is not None:
                results.append(result)

        return results

    def process_components(self, image, prediction, components, index, degree, line_height, baseline_resolution, with_image=True):
        pass