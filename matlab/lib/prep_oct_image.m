function I = prep_oct_image(f, outSize)
%PREP_OCT_IMAGE  Read an OCT image exactly as the original code did:
%   grey -> 3 identical channels, then resize to 256 x 256 (uint8).
%   The network-specific resize (256 -> 224 for DenseNet201) is done by
%   augmentedImageDatastore, as in the original code.
if nargin < 2, outSize = [256 256]; end
I = imread(f);
if size(I,3) == 1
    I = repmat(I, 1, 1, 3);
elseif size(I,3) > 3
    I = I(:,:,1:3);
end
I = imresize(I, outSize);
end
