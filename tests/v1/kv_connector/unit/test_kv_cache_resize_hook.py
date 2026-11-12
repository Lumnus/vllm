# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which KV connectors follow a KV cache that grows while serving."""

from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)


def test_connectors_refuse_a_resize_unless_they_opt_in():
    connector = MagicMock(spec=KVConnectorBase_V1)
    assert KVConnectorBase_V1.supports_kv_cache_resize(connector) is False
    with pytest.raises(NotImplementedError, match="does not support resizing"):
        KVConnectorBase_V1.resize_kv_caches(connector, {}, 8)


def test_offloading_connector_hands_the_resize_to_its_worker():
    connector = object.__new__(OffloadingConnector)
    connector.connector_worker = MagicMock()
    assert connector.supports_kv_cache_resize()
    kv_caches = {"layer": MagicMock()}
    connector.resize_kv_caches(kv_caches, 12)
    connector.connector_worker.resize_kv_caches.assert_called_once_with(kv_caches, 12)


@pytest.mark.parametrize(
    ("supports", "expected"),
    [((True, True), True), ((True, False), False), ((False, False), False)],
)
def test_multi_connector_supports_a_resize_only_if_all_children_do(supports, expected):
    multi = object.__new__(MultiConnector)
    multi._connectors = [
        MagicMock(supports_kv_cache_resize=MagicMock(return_value=s)) for s in supports
    ]
    assert multi.supports_kv_cache_resize() is expected


def test_multi_connector_resizes_every_child():
    multi = object.__new__(MultiConnector)
    multi._connectors = [MagicMock(), MagicMock()]
    kv_caches = {"layer": MagicMock()}
    multi.resize_kv_caches(kv_caches, 12)
    for child in multi._connectors:
        child.resize_kv_caches.assert_called_once_with(kv_caches, 12)
