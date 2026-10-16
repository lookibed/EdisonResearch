"""Trusted bit-transform adapters, shared by component and system experiments."""
from typing import Protocol
from .world import encode, decode


def valid_bits(value):
    return type(value) is list and all(type(b) is int and b in (0, 1) for b in value)


class BitTransform(Protocol):
    def encoded_length(self, length: int) -> int: ...
    def encode(self, bits: list[int]) -> list[int]: ...
    def decode(self, bits: list[int], original_length: int) -> list[int]: ...


class IdentityTransform:
    def encoded_length(self, length):
        return length

    def encode(self, bits):
        return list(bits)

    def decode(self, bits, original_length):
        return list(bits[:original_length])


class HammingTransform:
    def encoded_length(self, length):
        return ((length + 3) // 4) * 7

    def encode(self, bits):
        padded = list(bits) + [0] * (-len(bits) % 4)
        return [bit for i in range(0, len(padded), 4) for bit in encode(padded[i:i + 4], 'hamming')]

    def decode(self, bits, original_length):
        return [bit for i in range(0, len(bits), 7)
                for bit in decode(bits[i:i + 7], 'hamming')][:original_length]
