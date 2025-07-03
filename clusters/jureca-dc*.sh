# Environment setup for JURECA-DC cluster
echo "Setting up environment for JURECA-DC..."

# Set environment variables directly
export CLUSTER_NAME="jureca-dc"
export PROJECT_ID="EXAMPLE_PROJECT_123"
export SOFTWARE_STACK="/jureca/software"

# Example of a post-command
# echo "Setting up user environment on JURECA-DC..."
# ulimit -s unlimited

# Informative message for the user
echo "Welcome to the JURECA-DC cluster. Your project ID is set to \$PROJECT_ID." 