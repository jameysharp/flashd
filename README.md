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

Here's a sample resource, illustrating use of most of the schema.

- This example puts one representation inline instead of in a separate
  file, which is good as long as the response is very small and is valid
  UTF-8.
- The other representations give filenames for the actual response data.
  Constructing those files is left as an exercise to the reader.
- Note that the `header_length` includes the `"\r\n"` that terminates
  the header section, and remember that every representation must
  include its headers (if any) and that blank line before the body.
- The `status` field defaults to 200 if not specified, but I've made it
  explicit in this example.
- If you have an `ETag` header in a representation, you should copy its
  value into the metadata so the server can correctly handle conditional
  requests.
- If you have multiple representations which are subject to content
  negotiation, you need to tell the server which request headers should
  select each representation. The header names must include a trailing
  colon (":"). Also, for correct HTTP cache behavior, make sure each
  representation gets a different ETag.

Please see the documentation in `resource.fbs` for details.

```json
{
  "representations": [
    {
      "status": 200,
      "etag": "\"ecf701f727d9e2d77c4aa49ac6fbbcc997278aca010bddeeb961c10cf54d435a\"",
      "source_type": "InlineSource",
      "source": {
        "contents": "Content-Type: text/plain; charset=us-ascii\r\nETag: \"ecf701f727d9e2d77c4aa49ac6fbbcc997278aca010bddeeb961c10cf54d435a\"\r\n\r\nhello world!\n"
      },
      "header_length": 120
    },
    {
      "status": 200,
      "etag": "\"597c0bff392438fe7398d9c936d75b35b5fe17d517350f067b51248761659287\"",
      "source_type": "FileSource",
      "source": {
        "filename": "sample/sample.html"
      },
      "header_length": 119
    },
    {
      "status": 200,
      "etag": "\"3521640924b7f2dae82ecea6130ca67ae5efea5a0dabe0b802898eaa2cb3edd3\"",
      "source_type": "FileSource",
      "source": {
        "filename": "sample/sample.html.gz"
      },
      "header_length": 143
    }
  ],
  "negotiations": [
    {
      "header": "accept:",
      "wildcard": "*/*",
      "must_match": true,
      "choices": [
        {
          "name": "text/plain",
          "specificity": 2,
          "representations": [0]
        },
        {
          "name": "text/html",
          "specificity": 2,
          "representations": [1, 2]
        },
        {
          "name": "text/*",
          "specificity": 1,
          "representations": [0, 1, 2]
        }
      ]
    },
    {
      "header": "accept-encoding:",
      "choices": [
        {
          "name": "identity",
          "representations": [0, 1]
        },
        {
          "name": "gzip",
          "representations": [2]
        }
      ]
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
one of the variants you configured. If you use a graphical web browser,
you'll probably get the compressed HTML version, while if you use curl
without any options, you'll get the plain-text "hello world!"

If the request target doesn't exist, the server will try to serve a
resource file named `error404`. If that also doesn't exist, it will
serve a compiled-in page, which is generated during the build process
from [`src/404.json`](src/404.json).
