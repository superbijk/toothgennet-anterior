#!/bin/bash

set -e

# Color codes for better UI
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# Activate conda environment if available
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate gen3d 2>/dev/null || true
fi
SESSION_DATA="$BASE_DIR/session_data"
CHECKPOINTS_DIR="$SESSION_DATA/checkpoints"
DATASET_DIR="$SESSION_DATA/dataset"
PRIORS_DIR="$SESSION_DATA/generated_priors"
OUTPUT_DIR="$BASE_DIR/outputs"

# Function to get the latest file matching a pattern
get_latest_file() {
    local dir=$1
    local pattern=$2
    local description=$3

    # Find files matching pattern, sort by modification time (newest first)
    local files=($(find "$dir" -maxdepth 1 -name "$pattern" -type f -printf "%T@ %p\n" | sort -n -r | cut -d' ' -f2-))

    if [ ${#files[@]} -eq 0 ]; then
        echo -e "${RED}No $description found in $dir${NC}" >&2
        return 1
    fi

    # Auto-select the first (newest) file
    local selected="${files[0]}"
    echo -e "${BLUE}Auto-selecting latest $description:${NC} ${GREEN}$(basename "$selected")${NC}" >&2
    echo "$selected"
    return 0
}

# Interactive file selector with arrow keys
select_file_interactive() {
    local dir=$1
    local pattern=$2
    local description=$3

    # Find files matching pattern, sort by name
    local files=($(find "$dir" -maxdepth 1 -name "$pattern" -type f | sort))

    if [ ${#files[@]} -eq 0 ]; then
        echo -e "${RED}No $description found in $dir${NC}" >&2
        return 1
    fi

    local selected=0
    local key=""

    # Hide cursor
    tput civis

    while true; do
        # Clear and redraw menu
        clear
        echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
        echo -e "${BLUE}║    Select ${description}                 ║${NC}"
        echo -e "${BLUE}╚════════════════════════════════════════╝${NC}\n"

        for i in "${!files[@]}"; do
            local filename=$(basename "${files[$i]}")
            if [ $i -eq $selected ]; then
                echo -e "${GREEN}▶ $filename${NC}"
            else
                echo -e "  $filename"
            fi
        done

        echo -e "\n${YELLOW}Use ↑/↓ arrows to navigate, Enter to select, q to quit${NC}"

        # Read single key
        read -rsn1 key

        case "$key" in
            $'\x1b')  # ESC sequence
                read -rsn2 key  # Read 2 more chars
                case "$key" in
                    '[A')  # Up arrow
                        ((selected--))
                        if [ $selected -lt 0 ]; then
                            selected=$((${#files[@]} - 1))
                        fi
                        ;;
                    '[B')  # Down arrow
                        ((selected++))
                        if [ $selected -ge ${#files[@]} ]; then
                            selected=0
                        fi
                        ;;
                esac
                ;;
            '')  # Enter
                # Show cursor
                tput cnorm
                echo "${files[$selected]}"
                return 0
                ;;
            'q'|'Q')  # Quit
                tput cnorm
                return 1
                ;;
        esac
    done
}

# Function to run metrics evaluation with number selection
run_metrics_evaluation() {
    clear
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║         Select Checkpoint              ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}\n"

    # Find all checkpoints
    local files=($(find "$CHECKPOINTS_DIR" -maxdepth 1 -name "*.pt" -type f | sort))

    if [ ${#files[@]} -eq 0 ]; then
        echo -e "${RED}No checkpoints found in $CHECKPOINTS_DIR${NC}"
        return 1
    fi

    # Show options
    echo -e "${GREEN}0.${NC} All Checkpoints"
    for i in "${!files[@]}"; do
        local num=$((i + 1))
        local filename=$(basename "${files[$i]}")
        echo -e "${GREEN}$num.${NC} $filename"
    done

    echo ""
    read -p "Enter number (or q to quit) [default: 0]: " choice

    # Handle quit
    if [[ "$choice" == "q" || "$choice" == "Q" ]]; then
        clear
        return 1
    fi

    # Default to 0 if empty
    if [[ -z "$choice" ]]; then
        choice=0
    fi

    # Validate number
    if ! [[ "$choice" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}Invalid input${NC}"
        sleep 1
        return 1
    fi

    clear

    # Handle selection
    if [ "$choice" -eq 0 ]; then
        # Selected "All Checkpoints" - run batch
        run_checkpoint_evaluation
        return $?
    elif [ "$choice" -ge 1 ] && [ "$choice" -le ${#files[@]} ]; then
        # Selected individual checkpoint - run single
        local checkpoint="${files[$((choice - 1))]}"
        local checkpoint_name=$(basename "$checkpoint" .pt)

        echo -e "${YELLOW}=== Metrics Evaluation ===${NC}\n"
        echo -e "${GREEN}Selected:${NC} $checkpoint_name\n"

        # Auto-select dataset
        dataset=$(get_latest_file "$DATASET_DIR" "*.pkl" "dataset")
        if [ $? -ne 0 ]; then return 1; fi

        echo -e "\n${YELLOW}Running evaluation...${NC}"
        echo -e "${BLUE}Output: $OUTPUT_DIR/metrics${NC}\n"

        # Run single checkpoint evaluation (unified script shows [1/1])
        python "$BASE_DIR/toothgennet/sources/metrics/evaluate.py" \
            --checkpoint_path "$checkpoint" \
            --dataset_path "$dataset" \
            --output_dir "$OUTPUT_DIR/metrics" \
            --n_sample_points 2048 \
            --batch_size 32 \
            --split val

        if [ $? -eq 0 ]; then
            echo -e "\n${GREEN}✓ Evaluation completed!${NC}"
            echo -e "  ${GREEN}outputs/metrics/$checkpoint_name/generated_samples.npy${NC}"
            echo -e "  ${GREEN}outputs/metrics/$checkpoint_name/evaluate_data.csv${NC}"
            echo -e "  ${GREEN}outputs/metrics/checkpoints_metric_data.csv${NC}"
        else
            echo -e "\n${RED}✗ Evaluation failed${NC}"
            return 1
        fi
    else
        echo -e "${RED}Invalid selection${NC}"
        sleep 1
        return 1
    fi
}



# Function to run checkpoint evaluation (auto-sequential with fixed priors)
run_checkpoint_evaluation() {
    echo -e "${YELLOW}=== Batch Checkpoint Evaluation ===${NC}\n"
    echo -e "${BLUE}Auto-sequential workflow:${NC}"
    echo -e "  ${BLUE}1. Check each checkpoint for existing evaluation${NC}"
    echo -e "  ${BLUE}2. Generate & evaluate (skip if exists)${NC}"
    echo -e "  ${BLUE}3. Update centralized metrics${NC}\n"

    # Check for fixed priors
    if [ ! -f "$PRIORS_DIR/sampled_prior.npy" ] || [ ! -f "$PRIORS_DIR/sampled_3d_gaussian.npy" ]; then
        echo -e "${RED}ERROR: Fixed priors not found!${NC}"
        echo -e "${YELLOW}Expected:${NC}"
        echo -e "  - $PRIORS_DIR/sampled_prior.npy"
        echo -e "  - $PRIORS_DIR/sampled_3d_gaussian.npy"
        return 1
    fi

    echo -e "${GREEN}✓ Found fixed priors${NC}\n"

    # Auto-select dataset
    dataset=$(get_latest_file "$DATASET_DIR" "*.pkl" "dataset")
    if [ $? -ne 0 ]; then return 1; fi

    echo -e "${YELLOW}Running evaluation...${NC}\n"

    # Run unified evaluate.py with checkpoint directory
    python "$BASE_DIR/toothgennet/sources/metrics/evaluate.py" \
        --checkpoint_path "$CHECKPOINTS_DIR" \
        --dataset_path "$dataset" \
        --output_dir "$OUTPUT_DIR/metrics" \
        --priors_dir "$PRIORS_DIR" \
        --n_sample_points 2048 \
        --batch_size 32 \
        --split val

    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ Complete!${NC}"
        echo -e "  ${GREEN}outputs/metrics/checkpoint-*/evaluate_data.csv${NC}"
        echo -e "  ${GREEN}outputs/metrics/checkpoints_metric_data.csv${NC}"
    else
        echo -e "\n${RED}✗ Failed${NC}"
        return 1
    fi
}

# Function to run tooth restoration
run_tooth_restoration() {
    echo -e "${YELLOW}=== Restore Damaged Tooth ===${NC}\n"

    # Auto-select checkpoint and dataset
    checkpoint=$(get_latest_file "$CHECKPOINTS_DIR" "*.pt" "checkpoint")
    if [ $? -ne 0 ]; then return 1; fi

    dataset=$(get_latest_file "$DATASET_DIR" "*.pkl" "dataset")
    if [ $? -ne 0 ]; then return 1; fi

    # Select cut type
    echo -e "${BLUE}Select Cut Type:${NC}"
    echo -e "  ${GREEN}1.${NC} Horizontal (gentle slope)"
    echo -e "  ${GREEN}2.${NC} Oblique (steep diagonal)"
    echo -e "  ${GREEN}3.${NC} Split (vertical plane)"
    read -p "Select cut type (default: 1): " cut_choice
    cut_choice=${cut_choice:-1}

    case $cut_choice in
        1) cut_type="horizontal" ;;
        2) cut_type="oblique" ;;
        3) cut_type="split" ;;
        *) cut_type="horizontal" ;;
    esac
    echo -e "${GREEN}Selected: $cut_type${NC}\n"

    # Get parameters
    read -p "Enter sample index to restore (leave empty for random): " sample_idx
    
    cmd_args="--checkpoint_path \"$checkpoint\" \
              --dataset_path \"$dataset\" \
              --cut_type $cut_type \
              --num_iterations 200 \
              --output_points 15000 \
              --seed 2024"

    if [ -z "$sample_idx" ]; then
        echo -e "${BLUE}Using random sample index${NC}"
        cmd_args="$cmd_args --random_sample"
    else
        cmd_args="$cmd_args --sample_idx $sample_idx"
    fi

    echo -e "\n${YELLOW}Running tooth restoration...${NC}"

    eval python "$BASE_DIR/toothgennet/sources/visualize/restore_tooth.py" $cmd_args

    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ Tooth restoration completed successfully!${NC}"
    else
        echo -e "\n${RED}✗ Tooth restoration failed.${NC}"
        return 1
    fi
}

# Function to run latent interpolation
run_latent_interpolation() {
    echo -e "${YELLOW}=== Latent Interpolation ===${NC}\n"

    # Auto-select checkpoint and dataset
    checkpoint=$(get_latest_file "$CHECKPOINTS_DIR" "*.pt" "checkpoint")
    if [ $? -ne 0 ]; then return 1; fi

    dataset=$(get_latest_file "$DATASET_DIR" "*.pkl" "dataset")
    if [ $? -ne 0 ]; then return 1; fi

    # Get parameters
    read -p "Enter start index (leave empty for random): " idx1
    idx1=${idx1:--1}

    read -p "Enter end index (leave empty for random): " idx2
    idx2=${idx2:--1}

    read -p "Enter number of steps (default: 10): " steps
    steps=${steps:-10}

    echo -e "\n${YELLOW}Running interpolation...${NC}"

    # Ensure PYTHONPATH is set
    export PYTHONPATH=$PYTHONPATH:.
    
    # No xvfb needed for matplotlib rendering
    python "$BASE_DIR/toothgennet/sources/visualize/interpolate.py" \
        --checkpoint_path "$checkpoint" \
        --dataset_path "$dataset" \
        --idx1 "$idx1" \
        --idx2 "$idx2" \
        --steps "$steps"

    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ Interpolation completed successfully!${NC}"
    else
        echo -e "\n${RED}✗ Interpolation failed.${NC}"
        return 1
    fi
}

# Function to start Flask viewer
start_viewer() {
    echo -e "${YELLOW}=== Starting Flask 3D Viewer ===${NC}"
    echo -e "Starting server at http://localhost:5000"

    # Ensure PYTHONPATH is set
    export PYTHONPATH=$PYTHONPATH:.
    
    python "$BASE_DIR/toothgennet/viewer/app.py"
}

# Main menu
main_menu() {
    clear
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║    ToothGenNet Publication Tools       ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}\n"

    echo -e "${GREEN}1.${NC} Evaluate Checkpoints"
    echo -e "${GREEN}2.${NC} Start Flask 3D Viewer"
    echo -e "${GREEN}3.${NC} Exit\n"

    read -n 1 -p "Select an option: " choice
    echo ""

    case $choice in
        1)
            run_metrics_evaluation
            ;;
        2)
            start_viewer
            ;;
        3)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option.${NC}"
            ;;
    esac

    # Return to menu
    echo -e "\n${YELLOW}Press Enter to return to main menu...${NC}"
    read
    main_menu
}

# Handle command line arguments
case "$1" in
    --viewer)
        start_viewer
        ;;
    *)
        main_menu
        ;;
esac

