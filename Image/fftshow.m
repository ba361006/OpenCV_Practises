function fftshow(f,type)

if nargin < 2
    type = 'log';
end

if(type == 'log')
    fl = log(1+abs(f));
    fm = max(fl(:));
    imshow(im2uint8(fl/fm))
elseif (type == 'abs')
    fa = abs(f);
    fm = max(fa(:));
    imshow(fa/fm)
else
    error('TYPE must be abs or log.');
end
    
        