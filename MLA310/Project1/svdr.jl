function svdr(X; tol = 1e-12);
# Compact SVD of input matrix X (https://en.wikipedia.org/wiki/Singular_value_decomposition#Compact_SVD)
# Outputs: The rk non-zero singular values (σ) and corressponding singular vectors (U, V)
# where rk is the rank of A.
    m,n = size(X); r = min(m,n);
    U, σ, V = svd(X);
    mtol = max(max(σ[1]*tol,tol),r*eps());
    ids = findall(σ .> mtol);
    U = U[:,ids]; σ = σ[ids]; V = V[:,ids];
    rk = ids[end];
    return U, σ, V, rk;
end
