module MLA310

using LinearAlgebra, VMLS, Statistics

# ------------------------------------------------------------------------------------------------

"""
    svdr(X)

Compact SVD of input matrix X. 

# Arguments
- `X`: matrix to be decomposed
- `tol=1e-12`: tolerance for rank calculation


# Returns
- `U`: left singular vectors
- `σ`: singular values
- `V`: right singular vectors
- `rk`: rank of A
"""
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
export svdr

# ------------------------------------------------------------------------------------------------

"""
    pvs(X, nsv)

Principal Variable Selection

# Arguments
- `X`: the datamatrix.
- `nsv`: the number of variables to be selected .
- `npc=minimum(size(X))`: number of PC's included in the voting (optional).

# Returns
- `Q`: orthonormal scores (associated with the selected variables).
- `R`: corresponding loadings. NOTE: R(:,vperm) is upper triangular.
- `ids`: indices arranged in the order of the nsv selected variables.
- `vperm`: indices arranged in the order of the nsv selected- and all non-selected variables. 
- `ssEX`: the partial variances explained by the selected variables.
- `ni`: the norms of the (residual) selected variables before the score-normalization (Q).
- `U`: the normalized PCA-scores.
- `σ`: singular values of the mean centered X
"""
function pvs(X, nsv; npc = minimum(size(X)))
# Q, R, ids, vperm, ssEX, ni, U, σ = pvs(X, nsv);

# INITIALIZATIONS:
x̄   = mean(X, dims=1); X = X .- x̄ ;      # Centering of X-data.
U, σ, ~, rk = svdr(X);                   # Essentially PCA of the centered X.
nsv  = min(nsv, rk);
ntol = 1e-10;                            # Threshold defining a vanishing norm.
m,n  = size(X);                          # Data dimensions.
ids  = 0*Array{Int64}(undef, nsv);       # Indices of the selected variables.
Q    = zeros(m,nsv); R = zeros(nsv,n);   # See Output-description above.
ssEX = zeros(nsv,1);                     # See Output-description above.
ni   = zeros(nsv,1);                     # See Output-description above.
vAll = zeros(1,n);                       # Vector for holding the votes for all variables.
npc  = max(min(npc,rk),2);               # At least 2 PC's are included in the voting function.
T    = U[:,1:npc].*σ[1:npc]';            # Non-normalized PCA-scores for implementing the voting.
ssTX = sum(σ.^2);                        # The total sum-of-squares.
k    = 1;  idn = 1:n;                    # Book-keeping: k-th variable to be selected & idn - indices of candidate variables available for voting/selection.
while k <= nsv
    vAll[idn] = sum((T'*X[:,idn]).^2, dims = 1)./sum(X[:,idn].^2, dims = 1); # Update the vAll-votes for the X-variables subject to selection.
    mx, si    = findmax(vAll); si = si[2];     # Identify the variable accounting for the maximum variance.
    ni[k]     = norm(X[:,si]);                 # norm of the selected candiate variable.
    if ni[k] > ntol                            # Consider only candidate variables with non-vanishing norms.
        ids[k]  = si;                          # Index of the a-th selected variable.
        ssEX[k] = 100*mx/ssTX;                 # Storing the fraction of variance not previously explained by the chosen variable.
        qk      = X[:,si]./ni[k];              # Unit vector in the direction of the selected variable.            
        R[k,idn]= (qk'*X[:,idn]);              # Deflation coefficients in the chosen (v) direction.        
        X[:,idn]= X[:,idn] - qk*R[[k],idn];    # Deflate X wrt the chosen variable (modified Gram-Schmidt step).
        Q[:,k]  = qk;                          # Store qa into Q.
        k       = k+1;                         # Update a to prepare for the subsequent selection.
    end
    idn = setdiff(idn,si);                  # Eliminate selected/vanishing variable from future voting assignments.
    vAll[si]=0;   #X[:,si] = 0;             # Set the future votes of the selected/vanishing variable to 0.
end
vperm = [ids; setdiff((1:n)', ids)];        # Indices of the selected- and unselected variables.
return Q, R, ids, vperm, ssEX, ni, U, σ;
end
export pvs

# ------------------------------------------------------------------------------------------------

"""
    my_PCA(X)

Principal Component Analysis

# Arguments
- `X`: matrix to be deflated
- `mc=2`: maximum components to extract

# Returns
- `T`: matrix of PCA-scores
- `σ²`: the PCA-variances
- `sσ²`: the total variance     
- `x̄`: row vector of the X-column mean values
- `V`: the PCA-loadings
- `σ`: the singular values
- `U`: the normalized PCA scores

# Examples
```julia-repl
julia> T, σ², sσ², x̄, V, σ, U = my_PCA([1 2; 2 2.5])
([-0.559016994374947; 0.5590169943749472;;], [0.6250000000000001], 0.6250000000000001, [1.5 2.25], [0.8944271909999159; 0.44721359549995787;;], [0.7905694150420949], [-0.707106781186547; 0.7071067811865472;;])
```
"""
function my_PCA(X; mc = 2)
    m,n = size(X)
    mc  = min(mc, min(n,m-1))  # Assure that the number of extracted components is consistent with the size of the problem.
    x̄  = mean(X, dims=1)
    U, σ, V = svd(X.-x̄); 
    σ²  = (σ.^2)./(m-1);  # the PCA-variances    
    sσ² = sum(σ²);        # the total variance     
    σ²  = σ²[1:mc];
    U   = U[:,1:mc]; σ = σ[1:mc]; V = V[:,1:mc]; # Restriction to the first mc components.
    T   = U.*(σ'./sqrt(m-1))   # the PCA-scores in scaled format (matching MultivariateStats)
    return T, σ², sσ², x̄, V, σ, U;
end
export my_PCA

# ------------------------------------------------------------------------------------------------

"""
    PCRk(X, y)

Principal Component Regression

# Arguments
- `X`: data-matrix
- `y`: response vector
- `k=1`: number of principal components

# Returns
- `β₀`: the constant terms (1×k vector) of the PCR-models for up to k components.
- `Β`: the PC regression coeffs (m×k matrix) for the X-data for up to k components.
"""
function PCRk(X, y, k = 1)
m,n = size(X)
mc  = min(k, min(n,m)-1) # Make sure that the maximum number of components is not too large.
x̄   = mean(X, dims=1) # - row vector of the X-column mean values.
ȳ   = mean(y, dims=1) # - the mean of the response values y.
y₀  = y.- ȳ;          # - the centered response vector.
U, σ, V = svd(X.-x̄)   # - SVD of the centered X-data to find the principal components.
U = U[:,1:k]; σ = σ[1:k]; V = V[:,1:k]; # Restriction to the first mc components.
q = (σ.^(-1).*(U'*y₀));   # the regression coeffs for the PCA-scores.
Β = cumsum(V.*q', dims=2) # the PCR-regression coeffs for the X-data based on up to mc components.
β₀ = ȳ .- x̄*Β;            # the correspondning constant terms for the PCR-models.
return β₀, Β;
end
export PCRk

# ------------------------------------------------------------------------------------------------

"""
    PLS_nip(X, y; mc = k)

The NIPALS algorithm for Partial Least Squares Regression (PLSR) with
data-matrix X and corresponding response vector y for models with
up to `mc` PLS components.

### Arguments
- `X` : data matrix of predictors.
- `y` : vector of responses.
- `mc`: the maximal number of PLS components to be calculated.

### Returns
- `β₀, Β`: the constant terms and regression coeffs for all PLS-models including up to `mc` components.
- `T`    : the matrix of normalized PLS-scores.
- `W`    : the matrix of normalized PLS-weights.
- `P`    : the PLS X-projection loadings.
- `q`    : the PLS y-projection loadings.

### Reference 
[Björck and Indahl (2017): Fast and stable partial least squares modelling: A benchmark study with theoretical comments]
(https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/cem.2898)

"""
function PLS_nip(X, y; mc = 2)
# β₀, Β, T, W, P, q = PLS_nip(X,y; mc = 3)
m,n = size(X)
mc  = min(mc, min(n,m)-1) # Assure that the number of extracted components is consistent with the size of the problem.
T   = zeros(m,mc); W   = zeros(n,mc); P   = zeros(n,mc)
q   = zeros(1,mc);        # - the regression coeffs for the PLS-scores.
x̄   = mean(X, dims=1)     # - row vector of the X-column mean values.
ȳ   = mean(y, dims=1)[1]  # - the mean of the response values y.
y   = y.- ȳ;              # - the centered response vector.
X   = X.- x̄;              # - the centered X-data
for a = 1:mc
    w = X'y;  w = w/norm(w);  W[:,a] = w;
    t = X*w;  t = t/norm(t);  T[:,a] = t;
# ------------------- Deflate X and y ----------------------
    P[:,a] = X't;         X = X - t*P[:,a]';
    q[a]   = (y't)[1];    y = y - q[a].*t;
end
# ---------- Calculate regression coefficients -------------
Β = cumsum((W/triu(P'W)).*q, dims = 2); # the PLS-regression coeffs for the X-data based on up to mc components.
#Β  = cumsum((W/Bidiagonal(P'W, :U)).*q, dims = 2); # the PLS-regression coeffs for the X-data based on up to mc components.
β₀ = ȳ .- x̄*Β;            # the correspondning constant terms for the PLS-models.
return β₀, Β, T, W, P, q;
end
export PLS_nip

# ------------------------------------------------------------------------------------------------
function plsHy(X, y; mc = 2)
    
    m,n = size(X);
    
    B = zeros(mc,2) # B Stored by diagonals
    W = zeros(n,mc);
    T = zeros(m,mc);
    q   = zeros(1,mc);
    β = zeros(n,mc);
    
    x̄   = mean(X, dims=1)     # - row vector of the X-column mean values.
    ȳ   = mean(y, dims=1)[1]  # - the mean of the response values y.
    y   = y.- ȳ;              # - the centered response vector.
    X   = X.- x̄;              # - the centered X-data

    w0 = X'*y; w = w0/norm(w0);
    
    t = X*w; rho = norm(t); t = t/rho; q[1] = (y'*t)[1];
    W[:,1] = w; T[:,1] = t; B[1,1] = rho;
    d = w/rho; β[:,1] = (t'*y).*d;
    y = y - t*q[1];

       for a = 2:mc
           w1 = X'*y; w = (w0-w1)/q[a-1] - rho*w; w0 = w1; # w = X'*t - rho*w;
        
           w = w - W[:,a-1]*(W[:,a-1]'*w); theta = norm(w); w = w/theta; # Reorthogonalize and normalize w
        
           t = X*w; t = t - T[:,a-1]*(T[:,a-1]'*t); #Reorthongalize 3 # (t=X*w - theta*t;)
        
           rho = norm(t); t = t/rho; q[a] = (y'*t)[1];
        
           W[:,a] = w; T[:,a] = t;
           B[a-1,2] = theta; B[a,1] = rho;

           d = (w-theta*d)/rho;

           β[:,a] = β[:,a-1] + q[a]*d;
           y = y-t*q[a];
       end
    
    β₀ = ȳ .- x̄*β;
    return β₀, β, B, T, W;#P, q;
    
end
export plsHy
# ------------------------------------------------------------------------------------------------
#=
"""
    lda(X, y; Xnew=X)

Linear Discriminant Analysis

### Arguments
- `X` : data matrix of predictors.
- `y` : vector of responses.
- `Xnew=X`: data matrix of predictors to be classified.

### Returns
- `class`: a vector of class labels for each sample in Xnew.
"""
function lda(X, y, Xnew)
    G = maximum(y); N = size(X)[1]; # Constants
    Y = Matrix(I(G))[y,:]; # Dummy matrix (integer lookup in identity matrix)
    μ = (Y'Y)\Y'X;         # The group centers (mean vectors for each group)
    Xc = X-μ[y,:];         # Centring of the samples according to the correct group centres
    Σ  = (Xc'Xc)./(N-G);   # Calculation of the common covariance matrix Σ
    priors = ones(size(Y,2))./size(Y,2) # Calculation of the empirical priors

    # Save these if you want to do the predictions in a separate function
    Σμ = Σ \ μ';                            # Intermediate 1
    μΣμ = diag(μ*Σμ) ;                      # Intermediate 2
    d = Xnew*Σμ .-0.5*μΣμ'.+ log.(priors'); # d_k(x) (without redundant terms)

    class = [argmax(d[i,:]) for i=1:size(d,1)]; # Argmax along the class discriminant values for each sample
#    class = mapslices(argmax,d,dims=2);        # Alternative formulation
    return(class)
end
export lda
=#
"""
    lda(X, y)

Linear Discriminant Analysis

### Arguments
- `X` : data matrix of predictors.
- `y` : vector of class labels (responses).

### Returns
- `d`: the LDA-classification function built from the (X,y)-data.
"""

function lda(X, y)
    G = maximum(y); N = size(X)[1]; # Constants
    Y = Matrix(I(G))[y,:]; # Dummy matrix (integer lookup in identity matrix)
    μ  = (Y'Y)\Y'X;        # The group centers (mean vectors for each group)
    Xc = X-μ[y,:];         # Centring of the samples according to the correct group centres
    Σ  = (Xc'Xc)./(N-G);   # Calculation of the common covariance matrix Σ
    Π  = vec(sum(Y, dims = 1)/size(Y,1)) # Set priors equal to the empirical group proportions              
    
    # Discriminant score function computing dₖ-values (without redundant terms) for the samples in some Xdata-matrix:
    Σμ = Σ \ μ';                            
    μΣμ = diag(μ*Σμ); 
    d(Xdata) = Xdata*Σμ .-0.5*μΣμ'.+ log.(Π'); # d_k(x) 
    
    return d
end
export lda

# ------------------------------------------------------------------------------------------------

function svm(X, y, λ)
    # Create variables for the separating hyperplane w'*x = b.
    w = Variable(size(X)[2])
    b = Variable()
    m = size(X)[1]
    # Form the objective.
    obj = λ * sumsquares(w) +
        1/m * sum(max(1 - y.*(X*w + b),0))
    # Form and solve problem.
    problem = minimize(obj)
    solve!(problem, SCS.Optimizer; silent_solver = true)
    return evaluate(w), evaluate(b)
end;
export svm

# ------------------------------------------------------------------------------------------------

"""
    RRsimple(X, y; λ = 0.1)

Simple ridge regression.

### Arguments
- `X` : data matrix of predictors.
- `y` : vector of responses.
- `λ` : the regularizarion parameter value.

### Returns
- `β₀, β`: the constant term and regression coeffs for the resulting RR model.

### Reference: 
[Ridge regression - from Wikipedia](https://en.wikipedia.org/wiki/Ridge_regression)


"""
function RRsimple(X, y; λ = 0.1)
x̄ = mean(X, dims = 1); ȳ = mean(y);
y  = y.- ȳ; X  = X.-x̄; # - Centering of the data.
β = (X'X + λ*I)\(X'y);
β₀ = ȳ .- x̄*β;
return β₀[1], β  # β₀[1] assures conversion into scalar type.
end
export RRsimple

# ------------------------------------------------------------------------------------------------

"""
    RRLooCV_qr(X,y,λ; ρ = 1:length(y))

Ridge regression modelling alternative with faster LooCV-calculations

### Arguments
- `X` : data matrix of predictors.
- `y` : vector of responses.
- `λ` : the regularizarion parameter value.
- `q` : indices of the samples to be held out one-by-one in the Cross-validation

### Returns
- `β₀, β, resₓᵥ, ŷₓᵥ` : the constant term, the regression coeffs, the LooCV residuals and the LooCV predictions.

### Reference: 
.............

"""
function RRLooCV_qr(X,y,λ; ρ = 1:length(y))
# β0, β, yhat, res, yhatcv, rescv, Q, R, h = rrLooCV_QRfast(X,y,λ,ρ = cvrows);
# Inputs: data X, responses y, reg. param. λ and ρ = cvrows (specification of LooCV-samples).
# Outputs: β0, β, ŷₓᵥ
mρ  = length(ρ); m,n = size(X);
x̄   = mean(X, dims=1); ȳ = mean(y, dims=1); # - mean of data
y  = y.- ȳ ; X  = X.-x̄;              # - centering data.
Q,R = qr([X; sqrt(λ)*I]);  Q = Matrix(Q)[1:m,:];
β   = R\Q'y; β₀ = ȳ - x̄*β;           # The RR regression coeffs.  
res = y[ρ] - X[ρ,:]*β;               # Residuals for the samples specified by ρ.
h   = sum(Q[ρ,:].^2, dims=2) .+ 1/m; # The leverage values for the ρ-specified samples.
resₓᵥ= res./(1 .- h);                # LooCV-residuals for the specified samples.
ŷₓᵥ = (y.+ ȳ) - resₓᵥ;               # LooCV-predictions for the specified samples.
return β₀[1], β, resₓᵥ, ŷₓᵥ;
end
export RRLooCV_qr

# ------------------------------------------------------------------------------------------------

"""
    TregsRLooCV(X, y, λₛ; smo = 0)

Ridge-/Tikhonov regression modelling alternative with faster LooCV-calculations for many

### Arguments
- `X` : data matrix of predictors.
- `y` : vector of responses.
- `λₛ` : a vector of regularizarion parameter values.
- `smo` : order of smoothing in regularization matrix L.

### Returns
- `PRESSλₛ, minid, λₘᵢₙ, βₘᵢₙ, hₘᵢₙ, U, σ, V` : see explanations in the comments below.

### Reference: 
.............

"""
function TregsRLooCV(X, y, λₛ; smo = 0) # THE INPUTs ARE: (X,y)-data, λs: a vector of reg. param. values, smo: order of smoothing.
m, n   = size(X); λₛ = reshape(λₛ, 1, length(λₛ))  # λₛ is a row vector of many reg. parameter values.
x̄     = mean(X, dims=1); ȳ = mean(y, dims=1);  y  = y.- ȳ;  X  = X.-x̄;  # mean and centering of (X,y)-data.
if smo > 0 # smo. the order of smoothing.
    L  = [speye(n); zeros(smo,n)]; for i = 1:smo L = diff(L, dims = 1); end
    X  = X/L; # L is a discrete derivative 'smoothing matrix' of order "smo".
end
U, σ, V = svdr(X);
σ_plus_λₛ_over_σ = (σ.+(λₛ./σ)); # SVD of X & (σ, λ)-factors required for calc. of β-coeffs and H.
β       = V*((U'*y)./σ_plus_λₛ_over_σ);             # Simultaneous calc. of the β-coeffs for all λₛ.
H       = (U.^2)*(σ./σ_plus_λₛ_over_σ).+1/m;        # Simultaneous calc. of the leverage vectors for all λₛ.
# PRESSλₛ   = sum(((y.-X*β)./(1 .-H)).^2, dims=1)'; # The PRESS-values corresponding to all λₛ.
PRESSλₛ  = sum(((y.-U*((σ.*(U'*y))./σ_plus_λₛ_over_σ))./(1 .-H)).^2, dims=1)'; # The PRESS-values corresponding to all λₛ.
minid   = argmin(PRESSλₛ)[1];      # Find index of minimum press-value & identify corresponding λ-value, ...
if smo  > 0  β = L\β; end         # The X-regression coeffs in cases of smoothing (smo > 0).
β  = [ȳ .- x̄*β; β]; βₘᵢₙ = β[:,minid]; λₘᵢₙ = λₛ[minid]; hₘᵢₙ = H[:,minid]; # ...regression coeffs including the constant term (βₘᵢₙ) and leverages (hₘᵢₙ).
return PRESSλₛ, minid, λₘᵢₙ, βₘᵢₙ, hₘᵢₙ, U, σ, V, β;     # The OUTPUT-parameters of the function.
end
export TregsRLooCV

# ------------------------------------------------------------------------------------------------

function kmeans_(x, k; maxiters = 100, tol = 1e-5)                                                                 #  1
    #Overriding the "error in method definition: function VMLS.kmeans must be explicitly imported to be extended"  
    #By writing kmeans_ instead of kmeans we display the current implementation                                           
    # INPUTS:  x - a set of vectors, k - the number of clusters to be calculated, maxiters - the maximum number 
    #          of iterations in the algorithm, tol - the (relative) tolerance limit for convergence.
    # OUTPUTs: c - cluster labels assigned for each sample, z - the (k) cluster centers, J - the objective 
    #          function values for each cluster iteration.
    
    N = length(x)     # Number of samples (vectors in x)                                                           #  2              #  4
    d = length(x[1])  # dimension of the sample vectors.                                                           #  3
    distances = zeros(N) # used to store the distance of each sample to the nearest representative.                #  4
                                                                                                                   #  5
    z = [zeros(d) for j=1:k] # used to store representatives.                                                      #  6  
    Js = zeros(maxiters);    # For storing the objective function values of the process                            #  7                                                                                 #  9
    # ’c’ is an array of N integers between 1 and k. The initial cluster assignment in c is chosen randomly.       #  8
    c = [ rand(1:k) for i in 1:N ]                                                                                 #  9 
                                                                                                                   # 10
    Jprevious = Inf # used in stopping condition                                                                   # 11
    for iter = 1:maxiters                                                                                          # 12
                                                                                                                   # 13
        # Cluster j representative is average of points in cluster j.                                              # 14
        for j = 1:k                                                                                                # 15
            cj = [i for i=1:N if c[i] == j] # find the indices of the samples associated with cluster j            # 16
            z[j] = sum(x[cj]) / length(cj); # the updated center of cluster j (the mean of the associated samples) # 17
        end;                                                                                                       # 18
                                                                                                                   # 19
        # For each x[i], find distance to the nearest representative                                               # 20
        # and update its group index.                                                                              # 21
        for i = 1:N                                                                                                # 22
            (distances[i], c[i]) =                                                                                 # 23
                findmin([norm(x[i] - z[j]) for j = 1:k])                                                           # 24
        end;                                                                                                       # 25
                                                                                                                   # 26
        # Compute clustering objective value of the current clustering.                                            # 27
        J = distances'*distances/N                                                                                 # 28
        Js[iter] = J;                                                                                              # 29
        # Show progress and terminate if J stopped decreasing or maximun number of iterations occur.               # 30
        #println("Iteration ", iter, ": Jclust = ", J, ".")                                                         # 31
        if (iter > 1 && abs(J - Jprevious) < tol * J) || iter == maxiters                                          # 32
            Js = Js[1:iter];                                                                                       # 33
            return c, z, Js                                                                                        # 34
        end                                                                                                        # 35
        Jprevious = J                                                                                              # 36
    end                                                                                                            # 37
                                                                                                                   # 38
end
export kmeans_

# ------------------------------------------------------------------------------------------------

function unfold(A,i)
    # Permutation vector
    dims = [i; setdiff(1:ndims(A),i)]
    # Permute so mode i comes first
    A = permutedims(A, dims)
    # Vectorize each across first dimension
    return reshape(A, size(A,1),:)
end
export unfold

function fold(A,i,siz)
    # Reshape to permuted target sizes
    dims = tuple((size(A,1))...,siz[1:end .!= i]...)
    B = reshape(A, dims)
    # Permutation vector
    dims = [1]
    if i > 1
        dims = [2:i; dims]
    end
    if i < ndims(B)
        dims = [dims; i+1:ndims(B)]
    end
    # Permute to target size
    return permutedims(B, dims)
end
export fold

# ------------------------------------------------------------------------------------------------

# Function for i-mode tensor-matrix multiplication
function Tmul(A,X,i)
   siz = [i for i in size(A)] 
   siz[i] = size(X,1)
   return fold(X * unfold(A,i), i, siz)
end
export Tmul

# ------------------------------------------------------------------------------------------------

function svd3(A)
    # Compute the HOSVD of a 3-way tensor A
    U1,s1,v = svd(unfold(A,1))
    U2,s2,v = svd(unfold(A,2))
    U3,s3,v = svd(unfold(A,3))
    S = Tmul(Tmul(Tmul(A,U1',1),U2',2),U3',3)
    return [U1,U2,U3,S,s1,s2,s3]
end
export svd3

# ------------------------------------------------------------------------------------------------

Tinner(A,B) = sum(A.*B)
export Tinner

# ------------------------------------------------------------------------------------------------

Tnorm(A) = Tinner(A,A)^0.5
export Tnorm

# ------------------------------------------------------------------------------------------------

"""
    pvr(X, nsv)

Principal Variable Regression

# Arguments
- `X`: the datamatrix.
- `y`: The response vector.
- `nsv`: the number of variables to be selected.
- `npc=minimum(size(X))`: number of PC's included in the voting (optional).

# Returns
- `Q`: orthonormal scores (associated with the selected variables).
- `R`: corresponding loadings. NOTE: R(:,vperm) is upper triangular.
- `ids`: indices arranged in the order of the nsv selected variables.
- `vperm`: indices arranged in the order of the nsv selected- and all non-selected variables. 
- `ssEX`: the partial variances explained by the selected variables.
- `ni`: the norms of the (residual) selected variables before the score-normalization (Q).
- `U`: the normalized PCA-scores.
- `σ`: singular values of the mean centered X
"""
function pvr(X, y, nsv; npc = minimum(size(X)))
# Q, R, ids, vperm, ssEX, ni, U, σ = pvr(X, nsv);
# Lines that are different from PVS has #! at the end
# INITIALIZATIONS:
x̄   = mean(X, dims=1); X = X .- x̄ ;      # Centering of X-data.
ȳ   = mean(y, dims=1); y = y .- ȳ;       #! Centering of y-data.
U, σ, _, rk = svdr(X);                   # Essentially PCA of the centered X.
nsv  = min(nsv, rk);                     # The number of variables to be selected.
ntol = 1e-10;                            # Threshold defining a vanishing norm.
m,n  = size(X);                          # Data dimensions.
ids  = 0*Array{Int64}(undef, nsv);       # Indices of the selected variables.
Q    = zeros(m,nsv); R = zeros(nsv,n);   # See Output-description above.
ssEX = zeros(nsv,1);                     # See Output-description above.
ni   = zeros(nsv,1);                     # See Output-description above.
vAll = zeros(1,n);                       # Vector for holding the votes for all variables.
vAllx= zeros(1,n);                       #! Vector for holding the X-part of the votes
vAlly= zeros(1,n);                       #! Vector for holding the y-part of the votes
npc  = max(min(npc,rk),2);               # At least 2 PC's are included in the voting function.
T    = U[:,1:npc].*σ[1:npc]';            # Non-normalized PCA-scores for implementing the voting.
ssTX = sum(σ.^2);                        # The total sum-of-squares.
k    = 1;  idn = 1:n;                    # Book-keeping: k-th variable to be selected & idn - indices of candidate variables available for voting/selection.
while k <= nsv
    vAllx[idn] = sum((T'*X[:,idn]).^2, dims = 1)./sum(X[:,idn].^2, dims = 1); #! PVS criterion
    vAlly[idn] = (y' * X[:,idn]).^2 ./ sum(X[:,idn].^2, dims = 1); #! Explained y-variance
    vAll[idn]  = vAllx[idn] .* vAlly[idn]; #! PVR criterion, product of explained X-variance and y-variance
    mx, si    = findmax(vAll); si = si[2];     # Identify the variable accounting for the maximum variance.
    ni[k]     = norm(X[:,si]);                 # norm of the selected candiate variable.
    if ni[k] > ntol                            # Consider only candidate variables with non-vanishing norms.
        ids[k]  = si;                          # Index of the a-th selected variable.
        ssEX[k] = 100*mx/ssTX;                 # Storing the fraction of variance not previously explained by the chosen variable.
        qk      = X[:,si]./ni[k];              # Unit vector in the direction of the selected variable.            
        R[k,idn]= (qk'*X[:,idn]);              # Deflation coefficients in the chosen (v) direction.        
        X[:,idn]= X[:,idn] - qk*R[[k],idn];    # Deflate X wrt the chosen variable (modified Gram-Schmidt step).
        y       = y - qk * (qk' * y);          #! Deflate y wrt the chosen variable
        Q[:,k]  = qk;                          # Store qa into Q.
        k       = k+1;                         # Update a to prepare for the subsequent selection.
    end
    idn = setdiff(idn,si);                  # Eliminate selected/vanishing variable from future voting assignments.
    vAll[si]=0;   #X[:,si] = 0;             # Set the future votes of the selected/vanishing variable to 0.
    vAllx[si]=0; #!
    vAlly[si]=0; #!
end
vperm = [ids; setdiff((1:n)', ids)];        # Indices of the selected- and unselected variables.
return Q, R, ids, vperm, ssEX, ni, U, σ;
end
export pvr

# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------

end # End of module