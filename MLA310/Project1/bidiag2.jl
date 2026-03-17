function bidiag2(X, y; mc = 2)
m,n = size(X)
mc  = min(mc, min(n,m)-1)  # Assure that the number of extracted components is consistent with the size of the problem.
T   = zeros(m,mc); W   = zeros(n,mc);
β   = zeros(n,mc);    # - the regression coeffs
q   = zeros(1,mc);    # - the regression coeffs for the PLS-scores.
x̄   = mean(X, dims=1) # - row vector of the X-column mean values.
ȳ   = mean(y, dims=1) # - the mean of the response values y.
y   = y.- ȳ ;         # - the centered response vector.
X   = X.- x̄;         # - the centered X-data
B   = zeros(mc,2);    # - bidiagonal matrix stored by diagonals
# ----------------- Start bidiagonalization --------------------
w   = X'y; w = w/norm(w); W[:,1] = w;
t   = X*w;  ρ = norm(t);   t = t/ρ; T[:,1] = t;
B[1,1] = ρ;
d = (w/ρ); β[:,1] = (t'y)[1]*d; # Regression coeffs. for the first component
# ---------------- Continue bidiagonalization ------------------
for a = 2:mc
w = X't - ρ*w;  w = w - W[:,1:a-1]*(W[:,1:a-1]'w);  # Reorthogonalize w
θ = norm(w);    w = w/θ; W[:,a] = w;
t = X*w - θ*t;  t = t - T[:,1:a-1]*(T[:,1:a-1]'t);  # Reorthogonalize t
ρ = norm(t);    t = t/ρ; T[:,a] = t;
B[a-1,2] = θ; B[a,1] = ρ;
# --------------- Update regression coefficients ---------------
d = (w - θ.*d)/ρ; β[:,a] = β[:,a-1] + (t'y)[1]*d;
end

B  = Bidiagonal(B[:,1], B[1:mc-1,2], :U) # Convert B to bidiagonal form.
q  = y'T;
β₀ = ȳ .- x̄*β;      # The correspondning constant terms for the PLS-models.
return β₀, β, T, B, W, q;
end