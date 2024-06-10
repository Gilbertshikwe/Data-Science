# MongoDB Server Setup

This guide will walk you through the process of installing and starting the MongoDB server on your local machine.

## Prerequisites

Before proceeding, ensure that you have the following prerequisites installed:

- For Windows: Make sure you have a compatible version of Windows (7, 8, or 10) installed.
- For macOS: Make sure you have a compatible version of macOS (10.10 or later) installed.
- For Linux: Make sure you have a compatible version of your Linux distribution installed.

## Installation

### Windows

1. Download the MongoDB Community Server from the official website: https://www.mongodb.com/download-center/community
2. Choose the appropriate version for your system (e.g., Windows 64-bit x64) and download the `.msi` installer.
3. Run the downloaded `.msi` installer and follow the prompts to complete the installation.
4. Once the installation is complete, you should have MongoDB installed on your system.

### macOS

1. Download the MongoDB Community Server from the official website: https://www.mongodb.com/download-center/community
2. Choose the appropriate version for your system (e.g., macOS 64-bit x64) and download the `.tgz` package.
3. Extract the downloaded `.tgz` package to a desired location (e.g., `/usr/local/mongodb`).
4. Add the MongoDB binaries to your system's `PATH` environment variable by adding the following line to your shell configuration file (e.g., `~/.bash_profile`):
   ```
   export PATH=/usr/local/mongodb/bin:$PATH
   ```
5. Reload the shell configuration file: `source ~/.bash_profile`

### Linux

The installation process for Linux varies depending on your distribution. Refer to the official MongoDB documentation for detailed instructions: https://docs.mongodb.com/manual/administration/install-on-linux/

Here is a summarized process written as a README for installing and setting up MongoDB on a Linux system:

---

# MongoDB Installation and Setup Guide

This guide provides step-by-step instructions to install and set up MongoDB on a Linux system.

## Step 1: Install MongoDB Server

1. **Download MongoDB**: Visit the [MongoDB Community Download Page](https://www.mongodb.com/try/download/community).
2. **Select Version and Platform**: Choose the appropriate version for your operating system and download it.
3. **Follow Installation Instructions**: Follow the provided instructions to complete the installation for your specific OS.

## Step 2: Create the Data Directory

MongoDB requires a data directory to store its data.

1. **Create the Data Directory**:

   ```bash
   sudo mkdir -p /data/db
   ```

2. **Set Permissions for the Data Directory**:

   MongoDB needs appropriate permissions to access the data directory. Set the permissions by running:

   ```bash
   sudo chown -R $(whoami) /data/db
   ```

   This command changes the ownership of the `/data/db` directory and its contents to your current user account.

## Step 3: Start MongoDB Server

1. **Start MongoDB**:

   Start the MongoDB server by running the `mongod` command:

   ```bash
   mongod
   ```

Sure, here's a README section that explains how to stop the MongoDB server running on port 27017:

### Stopping the MongoDB Server

If you need to stop the MongoDB server that is running on port 27017, follow these steps:

1. **Find the Process ID (PID) of the Process Using Port 27017:**

   Open a terminal and run the following command to list the process ID (PID) of the process using port 27017:
   ```sh
   sudo lsof -i :27017
   ```

   This command will display the details of the process using port 27017. Look for the PID in the output. It will look something like this:
   ```
   COMMAND  PID  USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
   mongod   1234  mongodb  11u  IPv4  12345  0t0  TCP *:27017 (LISTEN)
   ```

   In this example, the PID is `1234`.

2. **Stop the Process:**

   Use the PID found in the previous step to stop the MongoDB server process. Replace `<PID>` with the actual process ID in the following command:
   ```sh
   sudo kill -9 <PID>
   ```

   For example, if the PID is `1234`, the command will be:
   ```sh
   sudo kill -9 1234
   ```

   This command forcibly stops the process running on port 27017.

### Note

Using `kill -9` is a forceful way to stop a process. It is generally recommended to use the proper shutdown command for MongoDB when possible to ensure data integrity:
1. Open another terminal window.
2. Connect to the MongoDB server using the MongoDB shell (`mongosh` or `mongo`).
   ```sh
   mongosh
   ```
   or for older versions:
   ```sh
   mongo
   ```
3. Use the `admin` database and issue the shutdown command:
   ```javascript
   use admin
   db.shutdownServer()
   ```

By following the proper shutdown procedure, you can ensure that MongoDB stops gracefully, maintaining data integrity.

## Troubleshooting

If you encounter issues starting MongoDB, follow these additional steps:

1. **Check for Running MongoDB Instances**:

   Use the following command to check if another MongoDB instance is already running and occupying the default port (27017):

   ```bash
   pgrep mongod
   ```

2. **Stop Running MongoDB Instances**:

   If you see any process IDs (PIDs), stop the running MongoDB instance using the `kill` command:

   ```bash
   sudo kill <PID>
   ```

   Replace `<PID>` with the actual process ID.

## Additional Resources

- [MongoDB Documentation](https://docs.mongodb.com/): For detailed information and advanced configurations.
- [MongoDB Community](https://community.mongodb.com/): For support and discussions.

---

By following these steps, you should be able to successfully install and set up MongoDB on your Linux system. If you encounter further issues, refer to the MongoDB documentation or seek assistance from the MongoDB community.
