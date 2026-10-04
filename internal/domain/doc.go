// Package domain holds the core business entities of the LLM gateway and the
// prompt evaluation suite, together with the ports (interfaces) that outer
// layers must implement.
//
// The package is the innermost ring of the Clean Architecture: it depends only
// on the standard library and knows nothing about HTTP, SQL, or any concrete
// LLM vendor SDK. Gateways, repositories and services depend on it, never the
// other way around.
package domain
