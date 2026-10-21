"""Minimal classic-NetCDF CDF2 header parser for the ModE-Sim range reader."""

from __future__ import annotations

import struct

import acquire_round67_mode_ra_allowed_modern_subset_v2 as cdf1


class Cursor64(cdf1.Cursor):
    def u64(self) -> int:
        return struct.unpack(">Q", self.take(8))[0]


def parse_header(payload: bytes) -> cdf1.ClassicHeader:
    cursor = Cursor64(payload)
    if cursor.take(4) != b"CDF\x02":
        raise RuntimeError("source-is-not-classic-netcdf-cdf2")
    num_records = cursor.u32()
    dim_tag = cursor.u32()
    dim_count = cursor.u32()
    if dim_tag == 0 and dim_count == 0:
        dimensions: dict[str, int | None] = {}
    elif dim_tag == 10:
        dimensions = {}
        for _ in range(dim_count):
            name = cursor.name()
            size = cursor.u32()
            dimensions[name] = None if size == 0 else size
    else:
        raise RuntimeError(f"unexpected-dimension-tag:{dim_tag}")
    global_attributes = cdf1.parse_attributes(cursor)
    var_tag = cursor.u32()
    var_count = cursor.u32()
    if var_tag != 11:
        raise RuntimeError(f"unexpected-variable-tag:{var_tag}")
    dim_names = list(dimensions)
    variables: dict[str, cdf1.Variable] = {}
    for _ in range(var_count):
        name = cursor.name()
        dimension_count = cursor.u32()
        dim_ids = [cursor.u32() for _ in range(dimension_count)]
        attributes = cdf1.parse_attributes(cursor)
        type_code = cursor.u32()
        vsize = cursor.u32()
        begin = cursor.u64()
        variables[name] = cdf1.Variable(
            name=name,
            dimensions=tuple(dim_names[index] for index in dim_ids),
            attributes=attributes,
            type_code=type_code,
            vsize=vsize,
            begin=begin,
        )
    return cdf1.ClassicHeader(
        num_records=num_records,
        dimensions=dimensions,
        global_attributes=global_attributes,
        variables=variables,
        parsed_bytes=cursor.offset,
    )
