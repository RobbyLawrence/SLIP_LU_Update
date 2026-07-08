#!/bin/bash

# Check for required arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <basis_file_path> <upper_bound_x>"
    echo "Example: $0 cycle_Bases/basis_k0_B.txt 1903"
    exit 1
fi

BASIS_FILE="$1"
UPPER_BOUND="$2"
DATA_FILE="simulation_results.csv"

# Validate that the basis file exists
if [ ! -f "$BASIS_FILE" ]; then
    echo "Error: Basis file '$BASIS_FILE' not found."
    exit 1
fi

# Validate that the upper bound is a number
if ! [[ "$UPPER_BOUND" =~ ^[0-9]+$ ]] || [ "$UPPER_BOUND" -le 0 ]; then
    echo "Error: Upper bound must be a positive integer."
    exit 1
fi

# Write the header to our CSV file.  nnz_change is (updated factors of B')
# minus (original factors of B); upd_minus_ref is (updated factors of B')
# minus (COLAMD refactorization of B') -- negative means the update stayed
# sparser than a fresh COLAMD refactor.
echo "x,speedup,nnz_change,upd_minus_ref" > "$DATA_FILE"

echo "Starting iterations 1 to $UPPER_BOUND using basis: $BASIS_FILE..."

# Loop through all values of x
for (( x=1; x<=UPPER_BOUND; x++ )); do
    # Print progress every 100 iterations or on the last iteration
    if (( x % 100 == 0 )) || [ "$x" -eq "$UPPER_BOUND" ]; then
        echo "Processing x = $x / $UPPER_BOUND..."
    fi

    # Run the command and capture the output
    output=$(./perm -R "$BASIS_FILE" "$x" 2>&1)

    # Extract the speedup multiplier (e.g., 2.89)
    speedup=$(echo "$output" | awk -F': ' '/update vs. refactorization:/ {print $2}' | tr -d 'x ')

    # Extract the absolute nonzero factor change (e.g., +52 or -10)
    nnz_change=$(echo "$output" | sed -n 's/.*changed the factor nonzeros by \([-+]*[0-9]*\).*/\1/p')

    # Extract the update-minus-refactor nonzero delta (signed)
    upd_minus_ref=$(echo "$output" | sed -n 's/.*update minus refactor nonzeros: \([-+]*[0-9]*\).*/\1/p')

    # If all values were successfully parsed, log them
    if [[ -n "$speedup" && -n "$nnz_change" && -n "$upd_minus_ref" ]]; then
        echo "$x,$speedup,$nnz_change,$upd_minus_ref" >> "$DATA_FILE"
    else
        echo "Warning: Failed to parse output for x = $x" >&2
    fi
done

echo "Runs complete! Processing statistics..."
echo "--------------------------------------------------"

# Inline Python script using only built-in standard libraries (no pandas needed)
python3 - <<EOF
import csv

data_file = "$DATA_FILE"
rows = []

try:
    with open(data_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'x': int(row['x']),
                'speedup': float(row['speedup']),
                'nnz_change': int(row['nnz_change']),
                'upd_minus_ref': int(row['upd_minus_ref'])
            })
except Exception as e:
    print(f"Error reading CSV data file: {e}")
    exit()

if not rows:
    print("No data collected.")
    exit()

# --- Speedup Calculations ---
total_speedup = sum(r['speedup'] for r in rows)
avg_speedup = total_speedup / len(rows)

max_speedup_row = max(rows, key=lambda r: r['speedup'])
min_speedup_row = min(rows, key=lambda r: r['speedup'])

# --- Sparsity Calculations ---
total_nnz = sum(r['nnz_change'] for r in rows)
avg_nnz = total_nnz / len(rows)

max_nnz_row = max(rows, key=lambda r: r['nnz_change'])
min_nnz_row = min(rows, key=lambda r: r['nnz_change'])

# --- Print Results ---
print("=== SPEEDUP STATISTICS (update vs. refactorization) ===")
print(f"Average Speedup:  {avg_speedup:.2f}x")
print(f"Highest Speedup:  {max_speedup_row['speedup']:.2f}x (at x = {max_speedup_row['x']})")
print(f"Lowest Speedup:   {min_speedup_row['speedup']:.2f}x (at x = {min_speedup_row['x']})")
print()
print("=== SPARSITY STATISTICS (updated - original factor nonzeros) ===")
print(f"Average Change:   {avg_nnz:+.2f} nonzeros")
print(f"Biggest Increase: {max_nnz_row['nnz_change']:+d} nonzeros (at x = {max_nnz_row['x']})")
print(f"Biggest Decrease: {min_nnz_row['nnz_change']:+d} nonzeros (at x = {min_nnz_row['x']})")
print()

# --- Update vs. Refactor Sparsity ---
total_umr = sum(r['upd_minus_ref'] for r in rows)
avg_umr = total_umr / len(rows)
max_umr_row = max(rows, key=lambda r: r['upd_minus_ref'])
min_umr_row = min(rows, key=lambda r: r['upd_minus_ref'])
sparser_count = sum(1 for r in rows if r['upd_minus_ref'] < 0)
denser_count  = sum(1 for r in rows if r['upd_minus_ref'] > 0)
same_count    = sum(1 for r in rows if r['upd_minus_ref'] == 0)

print("=== UPDATE vs. REFACTOR SPARSITY (update - refactor nonzeros) ===")
print(f"Average Delta:    {avg_umr:+.2f} nonzeros")
print(f"Update sparser:   {sparser_count} / {len(rows)} columns")
print(f"Update denser:    {denser_count} / {len(rows)} columns")
print(f"Update matches:   {same_count} / {len(rows)} columns")
print(f"Sparsest win:     {min_umr_row['upd_minus_ref']:+d} nonzeros (at x = {min_umr_row['x']})")
print(f"Densest loss:     {max_umr_row['upd_minus_ref']:+d} nonzeros (at x = {max_umr_row['x']})")
print("--------------------------------------------------")
print(f"Raw data saved securely to: {data_file}")
EOF
