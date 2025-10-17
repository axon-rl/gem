# NOTE: Vibecoded helper script.
# This has been tested to work only on ARM macOS.
# TODO: Verify and modify as needed for Linux.

set -e  # Exit on error

# Color codes for prettier output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEFAULT_VOLUME_DIR="kiwix_volume"
DEFAULT_LANGUAGE="en"
DEFAULT_VARIANT="nopic"
KIWIX_PORT=8080
DOWNLOAD_URL="https://download.kiwix.org/zim/wikipedia"

# Function to print colored messages
print_info() {
    printf "${BLUE}ℹ ${NC}$1\n" >&2
}

print_success() {
    printf "${GREEN}✓${NC} $1\n" >&2
}

print_warning() {
    printf "${YELLOW}⚠${NC} $1\n" >&2
}

print_error() {
    printf "${RED}✗${NC} $1\n" >&2
}

# Function to check if Docker is installed
check_docker() {
    if command -v docker &> /dev/null; then
        print_success "Docker is already installed"
        return 0
    else
        print_warning "Docker is not installed"
        return 1
    fi
}

# Function to detect OS and install Docker
install_docker() {
    print_info "Attempting to install Docker..."
    
    # Detect the operating system
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_info "Detected Linux system"
        
        # Check if we have sudo privileges
        if ! sudo -v &> /dev/null; then
            print_error "This script needs sudo privileges to install Docker"
            exit 1
        fi
        
        print_info "Installing Docker via official installation script..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        rm get-docker.sh
        
        # Add current user to docker group to run without sudo
        print_info "Adding current user to docker group..."
        sudo usermod -aG docker "$USER"
        
        print_success "Docker installed successfully!"
        print_warning "You may need to log out and back in for group membership to take effect"
        print_info "Or run: newgrp docker"
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_info "Detected macOS system"
        print_warning "Please install Docker Desktop manually from:"
        print_info "https://docs.docker.com/desktop/install/mac-install/"
        exit 1
        
    else
        print_error "Unsupported operating system: $OSTYPE"
        print_info "Please install Docker manually from https://docs.docker.com/get-docker/"
        exit 1
    fi
}

# Function to display language and variant information
show_options_info() {
    echo ""
    print_info "=== Wikipedia Language Options ==="
    echo "  Common languages:"
    echo "    en         - English (full Wikipedia, ~50GB nopic, ~100GB maxi)"
    echo "    simple     - Simple English (easier vocabulary, ~1GB nopic)"
    echo "    fr         - French"
    echo "    de         - German"
    echo "    es         - Spanish"
    echo "    ja         - Japanese"
    echo "    zh         - Chinese"
    echo ""
    print_info "  Find more language codes at: https://en.wikipedia.org/wiki/List_of_Wikipedias"
    echo ""
    print_info "=== Wikipedia Variants ==="
    echo "  maxi       - Full version with all images (~100GB+ for English)"
    echo "  nopic      - Complete articles without images (~50GB for English)"
    echo "  mini       - Only article introductions (~5GB for English)"
    echo ""
}

# Function to get the latest Wikipedia ZIM file for a given language and variant
get_latest_zim_filename() {
    local language="$1"
    local variant="$2"
    
    print_info "Finding the latest Wikipedia ${language} (${variant}) ZIM file..."
    
    # Build the pattern to search for
    # Pattern: wikipedia_{language}_all_{variant}_YYYY-MM.zim
    local pattern="wikipedia_${language}_all_${variant}_[0-9]{4}-[0-9]{2}\.zim"
    
    # Try to fetch the directory listing and parse for the latest file matching our pattern
    # Using -E for extended regex (portable across BSD and GNU grep)
    # [0-9] instead of \d for digit matching (POSIX-compliant)
    local latest_file=$(curl -s "$DOWNLOAD_URL/" | \
        grep -oE "wikipedia_${language}_all_${variant}_[0-9]{4}-[0-9]{2}\.zim" | \
        sort -u | \
        tail -1)
    
    if [ -z "$latest_file" ]; then
        print_error "Could not automatically detect the latest file for language '${language}' variant '${variant}'"
        print_warning "This might mean:"
        print_info "  - The language code is incorrect"
        print_info "  - This variant doesn't exist for this language"
        print_info "  - Network issues prevented fetching the file list"
        print_info "You can browse available files at: $DOWNLOAD_URL/"
        read -p "Enter the exact ZIM filename to download (or 'q' to quit): " manual_filename
        if [[ "$manual_filename" == "q" ]] || [[ -z "$manual_filename" ]]; then
            print_error "Cannot proceed without a valid filename. Exiting."
            exit 1
        fi
        echo "$manual_filename"
    else
        print_success "Found latest file: $latest_file"
        
        # Get file size estimate from the listing if possible
        # Using -E for extended regex (portable across BSD and GNU grep)
        local filesize=$(curl -s "$DOWNLOAD_URL/" | grep "$latest_file" | grep -oE '[0-9]+[KMG]' | tail -1)
        if [ -n "$filesize" ]; then
            print_info "Approximate size: $filesize"
        fi
        
        echo "$latest_file"
    fi
}

# Function to download the ZIM file
download_zim() {
    local volume_dir="$1"
    local zim_filename="$2"
    local zim_path="$volume_dir/$zim_filename"
    
    # Check if file already exists
    if [ -f "$zim_path" ]; then
        print_warning "ZIM file already exists at: $zim_path"
        read -p "Do you want to re-download it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping download, using existing file"
            return 0
        fi
    fi
    
    print_info "Downloading Wikipedia ZIM file..."
    print_warning "This may be a large file and will take a while!"
    print_info "Download URL: $DOWNLOAD_URL/$zim_filename"
    echo ""
    
    # Use wget if available (better for large files), otherwise curl
    if command -v wget &> /dev/null; then
        # -c enables continue/resume, --show-progress gives a nice progress bar
        wget -c --show-progress "$DOWNLOAD_URL/$zim_filename" -O "$zim_path"
    elif command -v curl &> /dev/null; then
        # -C - enables resume, -# shows progress bar
        curl -C - -L -# -o "$zim_path" "$DOWNLOAD_URL/$zim_filename"
    else
        print_error "Neither wget nor curl is available. Please install one of them."
        exit 1
    fi
    
    if [ $? -eq 0 ]; then
        echo ""
        print_success "Download completed!"
    else
        print_error "Download failed. Please check your internet connection and try again."
        print_info "The download can be resumed - just run this script again!"
        exit 1
    fi
}

# Main script execution
main() {
    echo ""
    print_info "=== Kiwix Wikipedia Offline Setup ==="
    echo ""
    
    # Step 1: Check for Docker
    if ! check_docker; then
        read -p "Would you like to install Docker now? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            print_error "Docker is required to run Kiwix. Exiting."
            exit 1
        fi
        install_docker
    fi
    
    # Verify Docker is working
    if ! docker ps &> /dev/null; then
        print_error "Docker is installed but not running or you don't have permissions"
        print_info "Try: sudo systemctl start docker"
        print_info "Or: newgrp docker"
        exit 1
    fi
    
    # Step 2: Choose language and variant
    echo ""
    show_options_info
    
    read -p "Enter Wikipedia language code (default: $DEFAULT_LANGUAGE): " language
    language="${language:-$DEFAULT_LANGUAGE}"
    language=$(echo "$language" | tr '[:upper:]' '[:lower:]')  # Convert to lowercase
    
    read -p "Enter variant (maxi/nopic/mini, default: $DEFAULT_VARIANT): " variant
    variant="${variant:-$DEFAULT_VARIANT}"
    variant=$(echo "$variant" | tr '[:upper:]' '[:lower:]')  # Convert to lowercase
    
    # Validate variant
    if [[ ! "$variant" =~ ^(maxi|nopic|mini)$ ]]; then
        print_warning "Invalid variant '$variant'. Using default: $DEFAULT_VARIANT"
        variant="$DEFAULT_VARIANT"
    fi
    
    print_success "Selected: Wikipedia ${language} (${variant} variant)"
    
    # Step 3: Set up volume directory
    echo ""
    print_info "Setting up storage directory for Wikipedia data..."
    read -p "Enter directory path (default: $DEFAULT_VOLUME_DIR): " volume_dir
    volume_dir="${volume_dir:-$DEFAULT_VOLUME_DIR}"
    
    # Convert to absolute path
    volume_dir=$(cd "$(dirname "$volume_dir")" 2>/dev/null && pwd)/$(basename "$volume_dir") || volume_dir="$(pwd)/$volume_dir"
    
    # Create directory if it doesn't exist
    mkdir -p "$volume_dir"
    print_success "Using directory: $volume_dir"
    
    # Step 4: Get latest ZIM filename for selected language and variant
    echo ""
    zim_filename=$(get_latest_zim_filename "$language" "$variant")
    
    # Step 5: Download ZIM file
    echo ""
    read -p "Download Wikipedia now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        download_zim "$volume_dir" "$zim_filename"
    else
        print_warning "Skipping download. Make sure you have a ZIM file in $volume_dir"
        print_info "You can download manually from: $DOWNLOAD_URL/"
    fi
    
    # Step 6: Start Kiwix container
    echo ""
    print_info "Starting Kiwix Docker container..."
    
    # Stop and remove existing container if it exists
    if docker ps -a --format '{{.Names}}' | grep -q "^kiwix-serve$"; then
        print_info "Removing existing kiwix-serve container..."
        docker stop kiwix-serve &> /dev/null || true
        docker rm kiwix-serve &> /dev/null || true
    fi
    
    # Start the container
    # The official Kiwix image is at ghcr.io/kiwix/kiwix-serve
    # We use *.zim to serve all ZIM files in the volume directory
    docker run -d \
        --name kiwix-serve \
        -v "$volume_dir:/data" \
        -p $KIWIX_PORT:8080 \
        --restart unless-stopped \
        ghcr.io/kiwix/kiwix-serve \
        "*.zim"
    
    if [ $? -eq 0 ]; then
        print_success "Kiwix container started successfully!"
        print_success "=== Setup Complete! ==="
        print_info "Access Wikipedia at: http://localhost:$KIWIX_PORT"
        print_info "Container name: kiwix-serve"
        print_info "Language: $language | Variant: $variant"
        print_info "Useful commands:"
        print_info "  Stop:    docker stop kiwix-serve"
        print_info "  Start:   docker start kiwix-serve"
        print_info "  Logs:    docker logs kiwix-serve"
        print_info "  Remove:  docker rm -f kiwix-serve"
        print_info "You can add more ZIM files to $volume_dir and restart the container to serve them all!"
    else
        print_error "Failed to start Kiwix container"
        exit 1
    fi
}

# Run main function
main