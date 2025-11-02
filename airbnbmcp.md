Airbnb Search & Listings - Desktop Extension (DXT)
A comprehensive Desktop Extension for searching Airbnb listings with advanced filtering capabilities and detailed property information retrieval. Built as a Model Context Protocol (MCP) server packaged in the Desktop Extension (DXT) format for easy installation and use with compatible AI applications.

Features
🔍 Advanced Search Capabilities
Location-based search with support for cities, states, and regions
Google Maps Place ID integration for precise location targeting
Date filtering with check-in and check-out date support
Guest configuration including adults, children, infants, and pets
Price range filtering with minimum and maximum price constraints
Pagination support for browsing through large result sets
🏠 Detailed Property Information
Comprehensive listing details including amenities, policies, and highlights
Location information with coordinates and neighborhood details
House rules and policies for informed booking decisions
Property descriptions and key features
Direct links to Airbnb listings for easy booking
🛡️ Security & Compliance
Robots.txt compliance with configurable override for testing
Request timeout management to prevent hanging requests
Enhanced error handling with detailed logging
Rate limiting awareness and respectful API usage
Secure configuration through DXT user settings
Installation
For Claude Desktop
This extension is packaged as a Desktop Extension (DXT) file. To install:

Download the .dxt file from the releases page
Open your compatible AI application (e.g., Claude Desktop)
Install the extension through the application's extension manager
Configure the extension settings as needed
For Cursor, etc.
Before starting make sure Node.js is installed on your desktop for npx to work.

Go to: Cursor Settings > Tools & Integrations > New MCP Server

Add one the following to your mcp.json:

{
  "mcpServers": {
    "airbnb": {
      "command": "npx",
      "args": [
        "-y",
        "@openbnb/mcp-server-airbnb"
      ]
    }
  }
}
To ignore robots.txt for all requests, use this version with --ignore-robots-txt args

{
  "mcpServers": {
    "airbnb": {
      "command": "npx",
      "args": [
        "-y",
        "@openbnb/mcp-server-airbnb",
        "--ignore-robots-txt"
      ]
    }
  }
}
Restart.

Configuration
The extension provides the following user-configurable options:

Ignore robots.txt
Type: Boolean (checkbox)
Default: false
Description: Bypass robots.txt restrictions when making requests to Airbnb
Recommendation: Keep disabled unless needed for testing purposes
Tools
airbnb_search
Search for Airbnb listings with comprehensive filtering options.

Parameters:

location (required): Location to search (e.g., "San Francisco, CA")
placeId (optional): Google Maps Place ID (overrides location)
checkin (optional): Check-in date in YYYY-MM-DD format
checkout (optional): Check-out date in YYYY-MM-DD format
adults (optional): Number of adults (default: 1)
children (optional): Number of children (default: 0)
infants (optional): Number of infants (default: 0)
pets (optional): Number of pets (default: 0)
minPrice (optional): Minimum price per night
maxPrice (optional): Maximum price per night
cursor (optional): Pagination cursor for browsing results
ignoreRobotsText (optional): Override robots.txt for this request
Returns:

Search results with property details, pricing, and direct links
Pagination information for browsing additional results
Search URL for reference
airbnb_listing_details
Get detailed information about a specific Airbnb listing.

Parameters:

id (required): Airbnb listing ID
checkin (optional): Check-in date in YYYY-MM-DD format
checkout (optional): Check-out date in YYYY-MM-DD format
adults (optional): Number of adults (default: 1)
children (optional): Number of children (default: 0)
infants (optional): Number of infants (default: 0)
pets (optional): Number of pets (default: 0)
ignoreRobotsText (optional): Override robots.txt for this request
Returns:

Detailed property information including:
Location details with coordinates
Amenities and facilities
House rules and policies
Property highlights and descriptions
Direct link to the listing
Technical Details
Architecture
Runtime: Node.js 18+
Protocol: Model Context Protocol (MCP) via stdio transport
Format: Desktop Extension (DXT) v0.1
Dependencies: Minimal external dependencies for security and reliability
Error Handling
Comprehensive error logging with timestamps
Graceful degradation when Airbnb's page structure changes
Timeout protection for network requests
Detailed error messages for troubleshooting
Security Measures
Robots.txt compliance by default
Request timeout limits
Input validation and sanitization
Secure environment variable handling
No sensitive data storage
Performance
Efficient HTML parsing with Cheerio
Request caching where appropriate
Minimal memory footprint
Fast startup and response times
Compatibility
Platforms: macOS, Windows, Linux
Node.js: 18.0.0 or higher
Claude Desktop: 0.10.0 or higher
Other MCP clients: Compatible with any MCP-supporting application
