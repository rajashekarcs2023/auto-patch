Vapi MCP Server

Copy page

Connect Vapi to AI assistants with Model Context Protocol (MCP)

Overview
The Vapi MCP Server exposes Vapi APIs as tools via the Model Context Protocol (MCP), so you can manage assistants, phone numbers, and calls from any MCP-compatible AI assistant (like Claude Desktop) or agent framework.

Use this server to connect your AI workflows to real-world telephony, automate voice tasks, and build richer conversational agents.

Looking to use MCP tools inside a Vapi assistant? See the MCP Tool documentation for integrating external MCP servers with your Vapi agents.

Using the Vapi CLI? Auto-configure MCP in your IDE with one command:

vapi mcp setup

This automatically configures Cursor, Windsurf, or VSCode with the Vapi MCP server. Learn more →

Quickstart: Claude Desktop Config
Fastest way to get started: connect Claude Desktop to the Vapi MCP Server.

1
Get your Vapi API key
Get your API key from the Vapi dashboard.

2
Edit Claude Desktop config
Open Settings → Developer tab → Edit Config.

3
Add the Vapi MCP server block
Insert this into your claude_desktop_config.json:

{
  "mcpServers": {
    "vapi-mcp": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://mcp.vapi.ai/mcp",
        "--header",
        "Authorization: Bearer ${VAPI_TOKEN}"
      ],
      "env": {
        "VAPI_TOKEN": "YOUR_VAPI_API_KEY"
      }
    }
  }
}

Replace YOUR_VAPI_API_KEY with your API key.

4
Restart Claude Desktop
Save and restart Claude Desktop.

Example prompt:

“Have my customer support assistant call Jeremy at +1555123456.”

Core Tools
The Vapi MCP Server exposes these actions as MCP tools:

Tool	Description	Example Usage
list_assistants	List all Vapi assistants	Show all configured assistants
create_assistant	Create a new Vapi assistant	Add a new assistant for a use case
get_assistant	Get a Vapi assistant by ID	View assistant config
list_calls	List all calls	Review call activity
create_call	Create an outbound call (now or scheduled)	Initiate or schedule a call
get_call	Get details for a specific call	Check status or result of a call
list_phone_numbers	List all Vapi phone numbers	See available numbers
get_phone_number	Get details of a specific phone number	Inspect a phone number
list_tools	List all available Vapi tools	Tool discovery
get_tool	Get details of a specific tool	Tool integration info
Scheduling calls: The create_call action supports scheduling with the optional scheduledAt parameter.

Integration Options
Remote (streamable-HTTP)
Remote (SSE)
OpenAI responses API
Local
Connect to the Vapi-hosted MCP server using the streamable-HTTP protocol.

Recommended for most production use cases.

Use this for clients or SDKs that support streamable-HTTP transport.

Endpoint: https://mcp.vapi.ai/mcp
Authentication: Pass your Vapi API key as a bearer token:
Authorization: Bearer YOUR_VAPI_API_KEY
Example config:

{
  "mcpServers": {
    "vapi-mcp": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://mcp.vapi.ai/mcp",
        "--header",
        "Authorization: Bearer ${VAPI_TOKEN}"
      ],
      "env": {
        "VAPI_TOKEN": "YOUR_VAPI_API_KEY"
      }
    }
  }
}

Custom MCP Client Integration
You can use any MCP-compatible client (SDKs available for multiple languages).

1
Install an MCP client SDK
Choose a language:

TypeScript
Python
Java
Kotlin
C#
2
Configure your connection
Set up your SDK to connect to the Vapi MCP Server (https://mcp.vapi.ai/sse) and authenticate with your API key.

3
Use MCP tools
Query available tools, list assistants, create calls, etc, via your SDK.

Example: Build a client with Node.js
Streamable-HTTP
SSE
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import dotenv from 'dotenv';
dotenv.config();
const mcpClient = new Client({ name: 'vapi-client', version: '1.0.0' });
const transport = new StreamableHTTPClientTransport(
  new URL('https://mcp.vapi.ai/mcp'),
  { requestInit: { headers: { Authorization: `Bearer ${process.env.VAPI_TOKEN}` } } }
);
async function main() {
  await mcpClient.connect(transport);
  const assistants = await mcpClient.callTool({ name: 'list_assistants', arguments: {} });
  console.log(assistants);
  await mcpClient.close();
}
main();

Detailed example: Build a client with Node.js
Streamable-HTTP
SSE
#!/usr/bin/env node
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import dotenv from 'dotenv';
// Load environment variables from .env file
dotenv.config();
// Ensure API key is available
if (!process.env.VAPI_TOKEN) {
  console.error('Error: VAPI_TOKEN environment variable is required');
  process.exit(1);
}
async function main() {
  try {
    // Initialize MCP client
    const mcpClient = new Client({
      name: 'vapi-client-example',
      version: '1.0.0',
    });
    // Create Streamable-HTTP transport for connection to remote Vapi MCP server
    const serverUrl = 'https://mcp.vapi.ai/mcp';
    const headers = {
      Authorization: `Bearer ${process.env.VAPI_TOKEN}`,
    };
    const options = {
      requestInit: { headers: headers },
    };
    const transport = new StreamableHTTPClientTransport(new URL(serverUrl), options);
    console.log('Connecting to Vapi MCP server via Streamable HTTP...');
    await mcpClient.connect(transport);
    console.log('Connected successfully');
    // Helper function to parse tool responses
    function parseToolResponse(response) {
      if (!response?.content) return response;
      const textItem = response.content.find(item => item.type === 'text');
      if (textItem?.text) {
        try {
          return JSON.parse(textItem.text);
        } catch {
          return textItem.text;
        }
      }
      return response;
    }
    try {
      // List available tools
      const toolsResult = await mcpClient.listTools();
      console.log('Available tools:');
      toolsResult.tools.forEach((tool) => {
        console.log(`- ${tool.name}: ${tool.description}`);
      });
      // List assistants
      console.log('\nListing assistants...');
      const assistantsResponse = await mcpClient.callTool({
        name: 'list_assistants',
        arguments: {},
      });
      const assistants = parseToolResponse(assistantsResponse);
      if (!(Array.isArray(assistants) && assistants.length > 0)) {
        console.log('No assistants found. Please create an assistant in the Vapi dashboard first.');
        return;
      }
      console.log('Your assistants:');
      assistants.forEach((assistant) => {
        console.log(`- ${assistant.name} (${assistant.id})`);
      });
      // List phone numbers
      console.log('\nListing phone numbers...');
      const phoneNumbersResponse = await mcpClient.callTool({
        name: 'list_phone_numbers',
        arguments: {},
      });
      const phoneNumbers = parseToolResponse(phoneNumbersResponse);
      if (!(Array.isArray(phoneNumbers) && phoneNumbers.length > 0)) {
        console.log('No phone numbers found. Please add a phone number in the Vapi dashboard first.');
        return;
      }
      console.log('Your phone numbers:');
      phoneNumbers.forEach((phoneNumber) => {
        console.log(`- ${phoneNumber.phoneNumber} (${phoneNumber.id})`);
      });
      // Create a call using the first assistant and first phone number
      const phoneNumberId = phoneNumbers[0].id;
      const assistantId = assistants[0].id;
      console.log(`\nCreating a call using assistant (${assistantId}) and phone number (${phoneNumberId})...`);
      const createCallResponse = await mcpClient.callTool({
        name: 'create_call',
        arguments: {
          assistantId: assistantId,
          phoneNumberId: phoneNumberId,
          customer: {
            number: "+1234567890"  // Replace with actual customer phone number
          }
          // Optional: schedule a call for the future
          // scheduledAt: "2025-04-15T15:30:00Z"
          // assistantOverrides: {
          //  variableValues: {
          //   name: 'John Doe',
          //   age: '25',
          //  },
          // },
        },
      });
      const createdCall = parseToolResponse(createCallResponse);
      console.log('Call created:', JSON.stringify(createdCall, null, 2));
    } finally {
      console.log('\nDisconnecting from server...');
      await mcpClient.close();
      console.log('Disconnected');
    }
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}
main();

For more detailed examples and complete client implementations, see the MCP Client Quickstart.
References