#!/usr/bin/env python3

"""
MCP Server for E2B Code Execution

A remote MCP server that exposes E2B code interpreter functionality
"""

import os
import json
import logging
import uuid
import zipfile
import io
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
import httpx

from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv
from supabase._async.client import AsyncClient as Client, create_client
from supabase.lib.client_options import ClientOptions

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
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Initialize Supabase async client (will be set during startup)
supabase_client: Optional[Client] = None


class ToolSchema(BaseModel):
    """Request model for code execution"""
    code: str = Field(..., description="Python code to execute in the E2B sandbox")
    file_urls: Optional[List[str]] = Field(None, description="Optional list of file URLs to download and make available in the sandbox")


# Initialize MCP server
mcp_server = Server("e2b-code-mcp-server")


async def download_file_from_url(url: str) -> bytes:
    """
    Download a file from a URL
    
    Args:
        url: URL of the file to download
        
    Returns:
        File content as bytes
        
    Raises:
        Exception: If download fails
    """
    try:
        logger.info(f"Downloading file from {url}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            logger.info(f"Successfully downloaded file from {url} ({len(response.content)} bytes)")
            return response.content
    except Exception as e:
        logger.error(f"Failed to download file from {url}: {str(e)}")
        raise Exception(f"Failed to download file from {url}: {str(e)}")


async def upload_to_supabase(filename: str, content: bytes) -> str:
    """
    Upload a file to Supabase storage and return a signed URL
    
    Args:
        filename: Name of the file (will be prefixed with UUID for uniqueness)
        content: File content as bytes
        
    Returns:
        Signed URL valid for 24 hours
        
    Raises:
        Exception: If upload fails or Supabase client is not configured
    """
    if not supabase_client:
        raise Exception("Supabase client not configured. Please set SUPABASE_URL and SUPABASE_KEY environment variables.")
    
    try:
        # Generate a unique filename to avoid collisions
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        # Determine content type based on file extension
        content_type = "application/octet-stream"
        if filename.lower().endswith('.zip'):
            content_type = "application/zip"
        
        logger.info(f"Uploading file {filename} to Supabase as {unique_filename}")
        
        # Upload to the 'exports' bucket (async)
        await supabase_client.storage.from_('exports').upload(
            path=unique_filename,
            file=content,
            file_options={"content-type": content_type}
        )
        
        # Generate signed URL valid for 24 hours (86400 seconds) (async)
        signed_url_response = await supabase_client.storage.from_('exports').create_signed_url(
            path=unique_filename,
            expires_in=86400
        )
        
        signed_url = signed_url_response['signedURL']
        logger.info(f"Successfully uploaded {filename} to Supabase: {signed_url}")
        
        return signed_url
        
    except Exception as e:
        logger.error(f"Failed to upload file {filename} to Supabase: {str(e)}")
        raise Exception(f"Failed to upload file to Supabase: {str(e)}")


@mcp_server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """
    List available tools for the MCP server
    """
    return [
        Tool(
            name="run_code",
            description="Run python code in a secure sandbox by E2B. Using the Jupyter Notebook syntax. Supports downloading files from URLs (file_urls parameter).",
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
    
    # Generate unique session ID for multi-tenant isolation
    session_id = str(uuid.uuid4())
    input_dir = f"/home/user/input_{session_id}"
    output_dir = f"/home/user/output_{session_id}"
    
    logger.info(f"Starting execution for session {session_id}")
    
    # Execute code in E2B sandbox
    sbx = None
    try:
        sbx = Sandbox()
        
        # Create input and output directories
        sbx.files.make_dir(input_dir)
        sbx.files.make_dir(output_dir)
        logger.info(f"Created directories: {input_dir}, {output_dir}")
        
        # Process file_urls if provided
        downloaded_files = []
        if arguments.file_urls:
            logger.info(f"Processing {len(arguments.file_urls)} file URLs")
            for file_url in arguments.file_urls:
                try:
                    # Download file
                    file_content = await download_file_from_url(file_url)
                    
                    # Extract filename from URL
                    filename = file_url.split('/')[-1].split('?')[0]
                    if not filename:
                        filename = f"file_{len(downloaded_files)}"
                    
                    # Upload to sandbox input directory
                    file_path = f"{input_dir}/{filename}"
                    sbx.files.write(file_path, file_content)
                    downloaded_files.append(filename)
                    logger.info(f"Uploaded {filename} to {file_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to process file URL {file_url}: {str(e)}")
                    # Continue with other files even if one fails
        
        # Execute the code
        execution = sbx.run_code(arguments.code)
        
        # Extract stdout and stderr - they could be strings, lists, or log objects
        stdout_output = execution.logs.stdout if hasattr(execution.logs, 'stdout') else []
        stderr_output = execution.logs.stderr if hasattr(execution.logs, 'stderr') else []
        
        # Handle different possible formats
        if isinstance(stdout_output, str):
            stdout_lines = [stdout_output] if stdout_output else []
        elif isinstance(stdout_output, list):
            # Check if list items have .line attribute or are already strings
            if stdout_output and hasattr(stdout_output[0], 'line'):
                stdout_lines = [log.line for log in stdout_output]
            else:
                stdout_lines = stdout_output
        else:
            stdout_lines = []
            
        if isinstance(stderr_output, str):
            stderr_lines = [stderr_output] if stderr_output else []
        elif isinstance(stderr_output, list):
            if stderr_output and hasattr(stderr_output[0], 'line'):
                stderr_lines = [log.line for log in stderr_output]
            else:
                stderr_lines = stderr_output
        else:
            stderr_lines = []
        
        logger.info(f"Execution completed: stdout={len(stdout_lines)} lines, stderr={len(stderr_lines)} lines")
        
        result = {
            "results": execution.results,  # Include execution results
            "stdout": stdout_lines,
            "stderr": stderr_lines,
            "status": "success",
            "session_id": session_id,
            "code": arguments.code,  # Include the executed code
            "input_directory": input_dir,
            "output_directory": output_dir,
            "downloaded_files": downloaded_files
        }
        
        # Check for generated files in output directory and create a zip file
        exported_zip = None
        if supabase_client:
            try:
                # List files in output directory
                output_files = sbx.files.list(output_dir)
                logger.info(f"Found {len(output_files)} files in output directory")
                
                # Filter for actual files (not directories)
                file_list = [f for f in output_files if f.type == "file"]
                
                if file_list:
                    # Create a zip file in memory
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for file_info in file_list:
                            try:
                                # Read file from sandbox
                                file_path = f"{output_dir}/{file_info.name}"
                                file_content = sbx.files.read(file_path)
                                
                                # Add file to zip
                                zip_file.writestr(file_info.name, file_content)
                                logger.info(f"Added {file_info.name} to zip archive")
                                
                            except Exception as e:
                                logger.error(f"Failed to add file {file_info.name} to zip: {str(e)}")
                                # Continue with other files even if one fails
                    
                    # Get the zip file bytes
                    zip_buffer.seek(0)
                    zip_content = zip_buffer.read()
                    
                    # Upload the zip file to Supabase
                    zip_filename = f"output_{session_id}.zip"
                    signed_url = await upload_to_supabase(zip_filename, zip_content)
                    exported_zip = {
                        "filename": zip_filename,
                        "url": signed_url,
                        "size": len(zip_content),
                        "files_count": len(file_list)
                    }
                    logger.info(f"Exported zip file with {len(file_list)} files to Supabase")
                else:
                    logger.info("No files found in output directory to export")
                            
            except Exception as e:
                logger.error(f"Failed to process output directory: {str(e)}")
                result["export_error"] = str(e)
        else:
            logger.info("Supabase not configured - skipping file export")
        
        if exported_zip:
            result["exported_zip"] = exported_zip
        
        # Close the sandbox
        if sbx:
            sbx.kill()
        
        return [
            TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )
        ]
        
    except Exception as e:
        error_message = f"Error executing code in E2B sandbox: {str(e)}"
        logger.error(error_message, exc_info=True)
        
        # Close the sandbox in case of error
        if sbx:
            try:
                sbx.kill()
            except:
                pass
        
        return [
            TextContent(
                type="text",
                text=json.dumps({
                    "error": error_message,
                    "status": "failed",
                    "session_id": session_id
                }, indent=2)
            )
        ]


# FastAPI app for HTTP transport
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI"""
    global supabase_client
    
    # Initialize Supabase async client if credentials are provided
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase_client = await create_client(
                SUPABASE_URL,
                SUPABASE_KEY,
                options=ClientOptions(
                    postgrest_client_timeout=10,
                    storage_client_timeout=10
                )
            )
            logger.info("Supabase async client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            supabase_client = None
    else:
        logger.warning("Supabase credentials not configured - file export functionality will be disabled")
    
    logger.info(f"Starting MCP E2B Code Server on {HOST}:{PORT}")
    logger.info(f"E2B API Key configured: {'Yes' if E2B_API_KEY else 'No'}")
    logger.info(f"Authentication enabled: {REQUIRE_AUTH}")
    logger.info(f"Secret Key configured: {'Yes' if SECRET_KEY else 'No'}")
    logger.info(f"Supabase configured: {'Yes' if supabase_client else 'No'}")
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
                "description": "Run python code in a secure sandbox. Using the Jupyter Notebook syntax. Supports downloading files from URLs (file_urls parameter) which is useful if you want to inspect files provided to you.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute in the E2B sandbox"
                        },
                        "file_urls": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Optional list of file URLs to download and make available in the sandbox"
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
