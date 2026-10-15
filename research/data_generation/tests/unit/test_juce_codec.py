"""Wire compatibility with the original bit-oriented JUCE state conversion."""

import random

import pytest

from obruxo_data.render.vita import JUCE_BASE64_ALPHABET, juce_memory_block_decode, juce_memory_block_encode


def reference_encode(value):
    characters = []
    for offset in range(0, len(value) * 8, 6):
        byte, shift = divmod(offset, 8)
        characters.append(JUCE_BASE64_ALPHABET[(int.from_bytes(value[byte:byte + 2], "little") >> shift) & 63])
    return f"{len(value)}.{''.join(characters)}"


def reference_decode(value):
    size, encoded = value.split(".", 1)
    output = bytearray(int(size))
    for index, character in enumerate(encoded):
        bits = JUCE_BASE64_ALPHABET.index(character)
        for bit in range(6):
            target = index * 6 + bit
            if target >= len(output) * 8:
                break
            if bits & (1 << bit):
                output[target // 8] |= 1 << (target % 8)
    return bytes(output)


def test_all_single_byte_encodings_match_original_wire_format():
    for value in range(256):
        payload = bytes([value])
        assert juce_memory_block_encode(payload) == reference_encode(payload)
        assert juce_memory_block_decode(reference_encode(payload)) == payload


def test_sizes_bit_boundaries_and_large_native_state_payload_match_reference():
    rng = random.Random(19)
    for size in [*range(129), 255, 256, 257, 1024, 256 * 1024]:
        payload = rng.randbytes(size)
        encoded = reference_encode(payload)
        assert juce_memory_block_encode(payload) == encoded
        assert juce_memory_block_decode(encoded) == payload


@pytest.mark.parametrize("value", ["0.", "0.ABC+", "1.A.", "1.A", "1.", "2.+", "4.ABC", "1.ABCDEFGH", "9.ABCDEFGHIJK"])
def test_declared_length_short_and_excess_bits_match_original(value):
    assert juce_memory_block_decode(value) == reference_decode(value)


@pytest.mark.parametrize("value", ["-1.", "x.A", "1.=", "1./", "0.?", "1.A?", "1.é", "1", "1.A\n"])
def test_invalid_encodings_remain_rejected(value):
    with pytest.raises(ValueError):
        juce_memory_block_decode(value)
