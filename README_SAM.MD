# SAMAPP - Getting Started Guide

Welcome to **SAMAPP**, an application designed for ease of use. This guide will help you set up and run the app on your MacBook Air. Follow the steps below carefully, and you’ll have the app running in no time!

---

## Requirements
1. **MacBook Air** with macOS installed.
2. **Internet connection** to download the necessary tools and files.

---

## Step-by-Step Instructions

### Step 1: Open Terminal

1. Click on the **Spotlight Search** (the magnifying glass in the top-right corner of your screen).
2. Type `Terminal` and hit **Enter**.
3. Terminal will open as a window with a command prompt.

---

### Step 2: Navigate to the samapp Folder

1. **If the samapp folder is on the Desktop**, type the following command and press **Enter**:

    ```bash
    cd ~/Desktop/samapp
    ```

2. **If the samapp folder is in Downloads**, type the following command and press **Enter**:

    ```bash
    cd ~/Downloads/samapp
    ```

---

### Step 3: Make the Script Executable

Once you're inside the `samapp` folder where the `run_samapp.sh` script is located, type the following command and press **Enter**:

```bash
chmod +x run_samapp.sh

Step 4: Run the Script
./run_samapp.sh


What the Script Does:
It checks if Docker is installed on your Mac.
If Docker is not installed, it will automatically download and install it for you.
It pulls the latest version of samapp from the internet.
It runs the samapp application and makes it available on your browser.

