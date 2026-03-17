using LinearAlgebra, Statistics, VMLS
include("svdr.jl");
function TregsRLooCV(X, y, λs; smo = 0) # THE INPUTs ARE: (X,y)-data, λs: a vector of reg. param. values, smo: order of smoothing.
m, n = size(X); λs = reshape(λs, 1, length(λs))  # λs is a row vector of many reg. parameter values.
x¯  = mean(X, dims=1); y¯  = mean(y, dims=1);    # - mean of (X,y)-data.
y   = y.- y¯;          X   = X.-x¯;              # - centering of (X,y)-data.
if smo > 0 # smo. the order of smoothing. 
    L = [speye(n); zeros(smo,n)]; for i = 1:smo L = diff(L, dims = 1); end
    X = X/L; # L is a discrete derivative smoothing matrix of order "smo".
end
U, σ, V = svdr(X); σ_plus_λs_over_σ = (σ.+(λs./σ));   # SVD of X & (σ, λ)-factors required for calc. of bcoefs and H.
bcoefs  = V*((U'*y)./σ_plus_λs_over_σ);              # Simultaneous calc. of the regression coeffs for all λs.
H       = (U.^2)*(σ./σ_plus_λs_over_σ).+1/m;         # Simultaneous calc. of the leverage vectors for all λs.
# press   = sum(((y.-X*bcoefs)./(1 .-H)).^2, dims=1)'; # The PRESS-values corresponding to all λs.
press   = sum(((y.-U*((σ.*(U'*y))./σ_plus_λs_over_σ))./(1 .-H)).^2, dims=1)'; # The PRESS-values corresponding to all λs.
if smo > 0  bcoefs = L\bcoefs; end                   # The X-regression coeffs in cases of smoothing (smo > 0).
minid = argmin(press)[1]; # Find index of minimum press-value & identify corresponding λ-value, ...
λ = λs[minid]; bλ = [y¯ .- x¯*bcoefs[:,minid]; bcoefs[:,minid]]; hλ = H[:,minid]; # ...regression coeffs (bλ) and leverages (hλ).
return press, minid, U, σ, V, H, bcoefs, λ, bλ, hλ;  # The OUTPUT-parameters of the function.
end
