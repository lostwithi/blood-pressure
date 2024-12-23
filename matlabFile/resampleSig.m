function [afterSig] = resampleSig(Signal,afterlength)
    afterlength = double(afterlength);
    afterSig = resample(Signal,afterlength,length(Signal));
end