% ---------------------------------------------------------------
% Driver for sDIPPushAndSwap (push K to end via diag / col pushes).
% ---------------------------------------------------------------

% --- Test 1: 5x5 pure diag chain, diagonally-dominant integer A ---
disp("======== Test 1: 5x5 pure diag chain, K=1 ========");
A1 = [ 5  1  0  2  0;
       0  4  1  0  1;
       1  0  5  0  0;
       0  1  0  6  1;
       0  0  1  0  4];
runTest(A1, 1);

% --- Test 2: 3x3, col push at k=1 (A(2,2) = 0) ---
disp(newline + "======== Test 2: 3x3 col push at k=1, K=1 ========");
A2 = [1 3 5;
      2 0 7;
      0 0 6];
runTest(A2, 1);

% --- Test 3: 5x5, col push at k=1, rest diag ---
disp(newline + "======== Test 3: 5x5 col push at k=1, K=1 ========");
A3 = [1  3  0  0  5;
      2  0  4  0  0;
      0  0  6  1  0;
      0  0  0  7  2;
      0  0  0  0  8];
runTest(A3, 1);

% --- Test 4: 6x6, col push at k=1, rest diag ---
disp(newline + "======== Test 4: 6x6 col push at k=1, K=1 ========");
A4 = [1  3  0  0  5  0;
      2  0  4  0  0  1;
      0  0  6  1  0  0;
      0  0  0  7  2  0;
      0  0  0  0  8  3;
      0  0  0  0  0  9];
runTest(A4, 1);

% --- Test 5: 6x6 pure diag chain, push from middle (K=3) ---
disp(newline + "======== Test 5: 6x6 pure diag chain, K=3 ========");
A5 = [ 5  1  0  2  0  0;
       0  4  1  0  2  0;
       0  0  3  1  0  2;
       0  0  1  6  0  0;
       0  0  0  0  5  1;
       0  0  0  0  0  4];
runTest(A5, 3);

% --- Test 6: 8x8 diagonally-dominant sparse, K=1 ---
disp(newline + "======== Test 6: 8x8 sparser diag chain, K=1 ========");
A6 = [ 3  1  0  0  2  0  0  0;
       0  4  2  0  0  1  0  0;
       1  0  5  1  0  0  0  0;
       0  2  0  6  0  0  1  0;
       0  0  1  0  4  0  0  2;
       0  0  0  1  0  5  0  0;
       0  0  0  0  2  0  3  1;
       0  0  0  0  0  1  0  6];
runTest(A6, 1);

% --- Test 7: 8x8 sparser, push from middle (K=4) ---
disp(newline + "======== Test 7: 8x8 sparser diag chain, K=4 ========");
runTest(A6, 4);

% --- Test 8: 6x6 with col push at k=1 forced by A(2,2)=0 ---
disp(newline + "======== Test 8: 6x6 col push at k=1, K=1 ========");
A8 = [2  3  1  0  0  1;
      1  0  2  0  0  0;
      0  1  4  1  0  0;
      0  0  1  5  1  0;
      0  0  0  1  6  1;
      1  0  0  0  1  4];
runTest(A8, 1);

% ---------------- helpers ----------------
function runTest(A, K)
    n = size(A, 1);

    % Guard: make sure A has an integer SLIP frame (no zero pivots).
    try
        origF = bareissFrame(A);
    catch err
        fprintf("*** SKIP: bareissFrame failed: %s\n", err.message);
        return;
    end
    if any(diag(origF) == 0)
        disp("*** SKIP: origF has a zero pivot on the diagonal.");
        return;
    end

    origA = lower(origF) * inv(constructD(origF)) * upper(origF);

    [Ftmp, P, Q] = sDIPPushAndSwap(K, origF);
    Fupdated = P * Ftmp * Q;
    permA    = P * origA * Q;

    err = computeError(permA, Fupdated);
    fprintf("computeError = %.3e\n", err);
    if err < 1e-8
        disp("*** PASS ***");
        disp("Ftmp:"); disp(Ftmp);
    else
        disp("*** FAIL ***");
        disp("Ftmp:");     disp(Ftmp);
        disp("Fupdated:"); disp(Fupdated);
        disp("P*origA*Q:"); disp(permA);
    end
end

% Compute the SLIP frame of A via Bareiss (integer-preserving).
% Errors out cleanly if a zero pivot is hit.
function F = bareissFrame(A)
    n = size(A, 1);
    F = zeros(n, n);
    Aprev = 1;
    Acur  = A;
    for k = 1:n
        if Acur(k, k) == 0
            error("bareissFrame:zeroPivot", "zero pivot at step k=%d", k);
        end
        F(k, k) = Acur(k, k);
        if k < n
            F(k+1:n, k) = Acur(k+1:n, k);
            F(k, k+1:n) = Acur(k, k+1:n);
            newA = Acur;
            for i = (k+1):n
                for j = (k+1):n
                    newA(i, j) = (Acur(k, k) * Acur(i, j) ...
                                  - Acur(i, k) * Acur(k, j)) / Aprev;
                end
            end
            Aprev = Acur(k, k);
            Acur  = newA;
        end
    end
end

function L = lower(F)
    n = size(F, 1);
    L = zeros(n, n);
    for i = 1:n
        for j = 1:i
            L(i, j) = F(i, j);
        end
    end
end

function U = upper(F)
    n = size(F, 1);
    U = zeros(n, n);
    for i = 1:n
        for j = i:n
            U(i, j) = F(i, j);
        end
    end
end

function D = constructD(F)
    n = size(F, 1);
    D = zeros(n, n);
    for i = 1:n
        if i == 1
            D(i, i) = F(i, i);
        else
            D(i, i) = F(i-1, i-1) * F(i, i);
        end
    end
end

function e = computeError(A, F)
    e = norm(A - lower(F) * (constructD(F) \ upper(F)));
end
