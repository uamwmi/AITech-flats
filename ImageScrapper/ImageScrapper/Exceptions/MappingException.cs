// SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
//
// SPDX-License-Identifier: MIT

using System.Runtime.Serialization;

namespace ResellScrapperV3.Exceptions;

[Serializable]
public class MappingException : Exception
{
    public MappingException() : base() { }
    public MappingException(string message) : base(message) { }
    public MappingException(string message, Exception inner) : base(message, inner) { }
    protected MappingException(SerializationInfo info, StreamingContext context) : base(info, context) { }
}