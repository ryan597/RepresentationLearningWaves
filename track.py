"""
Wave tracking from the segmentations provided through the leaned features.

We take in the segmentations from the WaveSeg model, we then obtain locations
for the segmentations, their centers, boundaries and if they are close to other
waves. The centers and boundaries will be used to identify waves from one image
to the next.
We also track whitecapping.
"""
