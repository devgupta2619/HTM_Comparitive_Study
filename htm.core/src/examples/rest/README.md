# SYNOPSIS 

This is an example REST web server and client implemented using a header-only package 
from [`cpp-httplib`](https://github.com/yhirose/cpp-httplib). This can be built and ran 
as a REST server to instantiate a Network class, configure it and run it as a NetworkAPI 
application.

This is a light-weight server.  If you need a full production REST server we suggest 
replacing this with one implemented from Boost.Beast.

# FEATURES

* Handles REST interface for htm.core library
* HTTP 1.0 / 1.1
* http

# DEPENDENCIES
   htm.core library

# USAGE

To run the server, 
  ./rest_server [port [network_interface]]
     port defaults to 8050
	 network_interface defaults to "127.0.0.1".
	 
To stop server,
  send a message:   /stop

To run the client,
  ./rest_client [ip_address [port]]
     ip_address defaults to "localhost".
	 port defaults to 8050
	 
You will find pre-built executables for both rest_server and rest_client in build/Release/bin.	 
For more details on how to build a client see [NetworkAPI_REST.md](../../docs/NetworkAPI_REST.md)

There is a Python version of the client in py/htm/examples/rest/client.py.
