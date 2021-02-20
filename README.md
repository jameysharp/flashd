flashd
======

There are lots of HTTP server implementations. This one's design goals
are:

- Serve static content only
- Provide complete control over response headers
- Configurability without a configuration language
- As little code as possible in the request path
- Standards-compliant implementation of content negotiation, conditional
  requests, and range requests

Other web servers use their own configuration languages for setting
response headers, mapping URLs to file paths, and declaring redirects.

This one instead requires you to do all that work ahead of time, using
any languages or tools you want, and compile the results into a simple
filesystem structure that allows the server to efficiently look up and
send the right response.

These pre-built responses include most of the response headers, with the
server only adding a few headers when they are required for correct
operation. In addition, you need to provide a little metadata so the
server never has to parse the headers you've provided, but can just copy
them directly to the client socket.

As a result:

- You can use any feature of HTTP, so long as you can pre-compute every
  response to any possible request. 

- The server does very little work for each request: after parsing the
  request headers, it can quickly open the right file and stream it out.

## Usage

I envision people building a variety of tools that make it easy to
construct appropriately-formatted responses. Until then, here's how it
works:

For each "request target" (which usually means a non-absolute URL, such
as `/` or `/robots.txt`), you can provide a "resource", which can have
one or more "representations". The process of deciding which
representation to send is called content negotiation.

A representation is the combination of response headers and body. Both
are optional, but there must be a blank line separating them, and
headers should use CRLF line endings, so the smallest valid
representation is `"\r\n"`.

Each representation should be in a separate file, which you may name
anything you want. You can reference the same representation from
multiple resources if you want to.

You then need to provide some metadata for the resource as a whole, in a
[FlatBuffers][]-encoded binary file that follows the schema in
[`src/resource.fbs`](src/resource.fbs). The simplest way to do that is
to write a JSON-encoded version of the metadata and then use `flatc -b
resource.fbs foo.json` to convert it to the binary representation.

[FlatBuffers]: https://google.github.io/flatbuffers/

Here's a trivial sample resource. This sample puts the representation
inline instead of in a separate file, which is good as long as the
response is very small and is valid UTF-8. Note that the `header_length`
includes the `"\r\n"` that terminates the header section, which in this
case is otherwise empty. The `status` field defaults to 200 if not
specified, but I've made it explicit in this example.

```json
{
  "representations": [
    {
      "source_type": "InlineSource",
      "source": {
        "contents": "\r\nhello world!\n"
      },
      "header_length": 2,
      "status": 200
    }
  ]
}
```

The binary resource metadata then needs to be named after the URL-safe
unpadded base-64 encoding of the sha-256 hash of the request target.
I've provided a shell script to help with that:

```
$ ./hash-target /
il7asoJjJEMhngUeSt4tHVu8Zxx4EFG_FDeJfL3-oPE
$ ./hash-target /robots.txt
ra2hMXUyoovr9j_m7b6j5D3GfYAtB8WDld5xszNqeaM
```

So if you name the binary version of the above sample
`il7asoJjJEMhngUeSt4tHVu8Zxx4EFG_FDeJfL3-oPE` and run `flashd` from the
same directory, then accessing `http://localhost:8080/` should return
"hello world!"

If the request target doesn't exist, the server will try to serve a
resource file named `error404`; if that also doesn't exist, it will
serve a compiled-in page converted from [`src/404.json`](src/404.json)
at build time.

There are a lot more options in the schema to support content
negotiation and conditional requests (most of which I haven't actually
implemented yet), so please see the documentation in `resource.fbs` for
details.
