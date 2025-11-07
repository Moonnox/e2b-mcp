#!/usr/bin/env python3

"""
MCP Server for E2B Code Execution

A remote MCP server that exposes E2B code interpreter functionality
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional
from collections.abc import Sequence

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from pydantic import BaseModel, Field, ValidationError
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
import uvicorn

from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
E2B_API_KEY = os.getenv("E2B_API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY", "")
PORT = int(os.getenv("PORT", "8080"))
HOST = os.getenv("HOST", "0.0.0.0")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() == "true"


class ToolSchema(BaseModel):
    """Request model for code execution"""
    code: str = Field(..., description="Python code to execute in the E2B sandbox")


# Initialize MCP server
mcp_server = Server("e2b-code-mcp-server")


@mcp_server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """
    List available tools for the MCP server
    """
    return [
        Tool(
            name="run_code",
            description="Run python code in a secure sandbox by E2B. Using the Jupyter Notebook syntax.",
            inputSchema=ToolSchema.model_json_schema()
        )
    ]


@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """
    Handle tool calls from the MCP client
    """
    if name != "run_code":
        raise ValueError(f"Unknown tool: {name}")
    
    if not arguments:
        raise ValueError("No arguments provided")
    
    try:
        arguments = ToolSchema.model_validate(arguments)
    except ValidationError as e:
        raise ValueError(f"Invalid code arguments: {e}") from e
    
    # Execute code in E2B sandbox
    try:
        sbx = Sandbox()
        execution = sbx.run_code(arguments.code)
        logger.info(f"Execution completed: stdout={len(execution.logs.stdout)} bytes, stderr={len(execution.logs.stderr)} bytes")
        
        result = {
            "stdout": execution.logs.stdout,
            "stderr": execution.logs.stderr,
            "status": "success"
        }
        
        return [
            TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )
        ]
        
    except Exception as e:
        error_message = f"Error executing code in E2B sandbox: {str(e)}"
        logger.error(error_message)
        return [
            TextContent(
                type="text",
                text=json.dumps({
                    "error": error_message,
                    "status": "failed"
                }, indent=2)
            )
        ]


# FastAPI app for HTTP transport
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI"""
    logger.info(f"Starting MCP E2B Code Server on {HOST}:{PORT}")
    logger.info(f"E2B API Key configured: {'Yes' if E2B_API_KEY else 'No'}")
    logger.info(f"Authentication enabled: {REQUIRE_AUTH}")
    logger.info(f"Secret Key configured: {'Yes' if SECRET_KEY else 'No'}")
    yield
    logger.info("Shutting down MCP E2B Code Server")


app = FastAPI(title="MCP E2B Code Server", lifespan=lifespan)
app.state.mcp = mcp_server


# Authentication Middleware
class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to verify the x-secret-key header for authentication.
    Only requires authentication for the tools/call method on /mcp endpoint.
    All other endpoints and methods are public for discovery.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth if not required (for local development)
        if not REQUIRE_AUTH:
            return await call_next(request)
        
        # Skip auth if SECRET_KEY is not configured
        if not SECRET_KEY:
            logger.warning("SECRET_KEY not configured but REQUIRE_AUTH is true")
            return await call_next(request)
        
        # Only check auth for /mcp endpoint with POST method
        if request.url.path == "/mcp" and request.method == "POST":
            try:
                # Read and store the body for reuse
                body_bytes = await request.body()
                
                # Parse JSON to check the method
                try:
                    body = json.loads(body_bytes)
                    method = body.get("method")
                    
                    # Only require auth for tools/call
                    if method == "tools/call":
                        # Check for x-secret-key header
                        provided_key = request.headers.get("x-secret-key")
                        
                        if not provided_key:
                            client_host = request.client.host if request.client else "unknown"
                            logger.warning(f"Missing x-secret-key header from {client_host} for tools/call")
                            return JSONResponse(
                                status_code=401,
                                content={
                                    "jsonrpc": "2.0",
                                    "error": {
                                        "code": -32001,
                                        "message": "Authentication required for tool execution"
                                    },
                                    "id": body.get("id")
                                }
                            )
                        
                        if provided_key != SECRET_KEY:
                            client_host = request.client.host if request.client else "unknown"
                            logger.warning(f"Invalid x-secret-key from {client_host} for tools/call")
                            return JSONResponse(
                                status_code=401,
                                content={
                                    "jsonrpc": "2.0",
                                    "error": {
                                        "code": -32001,
                                        "message": "Invalid authentication for tool execution"
                                    },
                                    "id": body.get("id")
                                }
                            )
                    
                    # For discovery methods (initialize, tools/list, resources/list, prompts/list), allow without auth
                    elif method in ["initialize", "tools/list", "resources/list", "prompts/list"]:
                        logger.debug(f"Allowing unauthenticated {method} request")
                    
                except json.JSONDecodeError:
                    # If we can't parse the JSON, let it through to be handled by the endpoint
                    logger.debug("Could not parse JSON body in auth middleware")
                
                # Create a new request with the body we read
                async def receive():
                    return {"type": "http.request", "body": body_bytes}
                
                request._receive = receive
                
            except Exception as e:
                logger.error(f"Error in auth middleware: {e}")
                # On error, let the request through to be handled properly
        
        # Continue with the request
        response = await call_next(request)
        return response


# Add the middleware to the app
app.add_middleware(AuthMiddleware)


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy", "service": "mcp-e2b-code"}


@app.post("/mcp")
async def handle_mcp_request(request: Request):
    """
    Main MCP endpoint for handling JSON-RPC requests
    """
    try:
        body = await request.json()
        
        # Log the incoming request for debugging
        logger.debug(f"Received MCP request: {json.dumps(body, indent=2)}")
        
        # Handle the MCP request
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")
        
        # Route to appropriate handler based on method
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "serverInfo": {
                    "name": "e2b-code-mcp-server",
                    "version": "1.0.0"
                }
            }
        elif method == "tools/list":
            tools = await handle_list_tools()
            result = {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema
                    }
                    for tool in tools
                ]
            }
        elif method == "resources/list":
            # Return empty resources list (this server uses tools, not resources)
            result = {
                "resources": []
            }
        elif method == "prompts/list":
            # Return empty prompts list (this server uses tools, not prompts)
            result = {
                "prompts": []
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_arguments = params.get("arguments", {})
            
            contents = await handle_call_tool(tool_name, tool_arguments)
            result = {
                "content": [
                    {
                        "type": content.type,
                        "text": content.text
                    }
                    for content in contents
                ]
            }
        else:
            # Method not found
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    },
                    "id": request_id
                },
                status_code=200
            )
        
        # Return successful response
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            },
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error handling MCP request: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                },
                "id": body.get("id") if "body" in locals() else None
            },
            status_code=200
        )


@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "service": "MCP E2B Code Server",
        "version": "1.0.0",
        "description": "Model Context Protocol server for E2B code execution",
        "endpoints": {
            "/health": "Health check endpoint",
            "/mcp": "MCP JSON-RPC endpoint",
            "/tools": "List available tools"
        }
    }


@app.get("/tools")
async def list_tools():
    """List available tools endpoint"""
    return {
        "tools": [
            {
                "name": "run_code",
                "description": "Run python code in a secure sandbox by E2B. Using the Jupyter Notebook syntax.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute in the E2B sandbox"
                        }
                    },
                    "required": ["code"]
                }
            }
        ]
    }


async def main():
    """Main entry point for the server"""
    # Run the server
    config = uvicorn.Config(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
