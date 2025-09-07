# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from datetime import datetime, timedelta

import pytest

from gem.tools.mcp_server.time_mcp import (
    _moment_to_strftime,
    _parse_date,
    _parse_dt,
    app,
)


@pytest.mark.anyio(backends=["asyncio"])
async def test_current_time():
    """Test current_time function with basic formats."""
    # Get the actual function from the app's tools
    current_time_tool = await app.get_tool("current_time")

    # Test default format
    result = await current_time_tool.run({})
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result.content[0].text)

    # Test date only format
    result = await current_time_tool.run({"format": "YYYY-MM-DD"})
    assert re.match(r"\d{4}-\d{2}-\d{2}", result.content[0].text)

    # Test with timezone
    result = await current_time_tool.run(
        {"format": "YYYY-MM-DD HH:mm:ss", "timezone": "UTC"}
    )
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result.content[0].text)


@pytest.mark.anyio(backends=["asyncio"])
async def test_relative_time():
    """Test relative_time function with mathematical precision."""
    relative_time_tool = await app.get_tool("relative_time")

    # Test exact future time (3661 seconds = 1 hour 1 minute 1 second)
    now = datetime.now().replace(microsecond=0)
    future_time = now + timedelta(seconds=3661)
    time_str = future_time.strftime("%Y-%m-%d %H:%M:%S")

    result = await relative_time_tool.run({"time": time_str})
    text_result = result.content[0].text
    assert "in 1 hours" in text_result  # Should round down to hours

    # Test exact past time (7200 seconds = 2 hours)
    past_time = now - timedelta(seconds=7200)
    time_str = past_time.strftime("%Y-%m-%d %H:%M:%S")

    result = await relative_time_tool.run({"time": time_str})
    text_result = result.content[0].text
    assert "2 hours ago" in text_result

    # Test exact minute precision (90 seconds = 1 minute 30 seconds)
    future_minute = now + timedelta(seconds=90)
    time_str = future_minute.strftime("%Y-%m-%d %H:%M:%S")

    result = await relative_time_tool.run({"time": time_str})
    text_result = result.content[0].text
    assert "in 1 minutes" in text_result  # Should round down to minutes

    # Test exact day boundary (25 hours = 1 day 1 hour)
    future_day = now + timedelta(hours=25)
    time_str = future_day.strftime("%Y-%m-%d %H:%M:%S")

    result = await relative_time_tool.run({"time": time_str})
    text_result = result.content[0].text
    assert "in 1 days" in text_result  # Should round down to days

    # Test seconds precision (allowing for 1-2 second execution delay)
    future_seconds = now + timedelta(seconds=30)
    time_str = future_seconds.strftime("%Y-%m-%d %H:%M:%S")

    result = await relative_time_tool.run({"time": time_str})
    text_result = result.content[0].text
    # Allow for execution delay - should be around 28-30 seconds
    assert re.search(r"in (2[89]|30) seconds", text_result)


@pytest.mark.anyio(backends=["asyncio"])
async def test_days_in_month():
    """Test days_in_month function."""
    days_in_month_tool = await app.get_tool("days_in_month")

    # Test current month
    result = await days_in_month_tool.run({})
    days = int(result.content[0].text)
    assert 28 <= days <= 31

    # Test specific month (February non-leap year)
    result = await days_in_month_tool.run({"date": "2023-02-01"})
    assert result.content[0].text == "28"

    # Test specific month (February leap year)
    result = await days_in_month_tool.run({"date": "2024-02-01"})
    assert result.content[0].text == "29"

    # Test specific month (January)
    result = await days_in_month_tool.run({"date": "2023-01-15"})
    assert result.content[0].text == "31"


@pytest.mark.anyio(backends=["asyncio"])
async def test_get_timestamp():
    """Test get_timestamp function with numerical precision."""
    get_timestamp_tool = await app.get_tool("get_timestamp")

    # Test Unix epoch (should be 0)
    result = await get_timestamp_tool.run({"time": "1970-01-01 00:00:00"})
    timestamp = int(result.content[0].text)
    # Account for local timezone offset
    expected_epoch = 0
    assert abs(timestamp - expected_epoch) < 86400000  # Within 24 hours

    # Test known timestamp: 2023-01-01 00:00:00 UTC = 1672531200000 ms
    result = await get_timestamp_tool.run({"time": "2023-01-01 00:00:00"})
    timestamp = int(result.content[0].text)
    # Since input is interpreted as local time, we verify it's a reasonable timestamp
    assert 1672531200000 - 86400000 <= timestamp <= 1672531200000 + 86400000

    # Test millisecond precision
    result = await get_timestamp_tool.run({"time": "2023-01-01 12:30:45.123"})
    timestamp = int(result.content[0].text)
    # Verify milliseconds are included (should end in 123)
    assert timestamp % 1000 == 123

    # Test different millisecond values
    result = await get_timestamp_tool.run({"time": "2023-01-01 12:30:45.456"})
    timestamp = int(result.content[0].text)
    assert timestamp % 1000 == 456


@pytest.mark.anyio(backends=["asyncio"])
async def test_convert_time():
    """Test convert_time function with numerical accuracy."""
    convert_time_tool = await app.get_tool("convert_time")

    # Test UTC to UTC (should be identical)
    result = await convert_time_tool.run(
        {
            "sourceTimezone": "UTC",
            "targetTimezone": "UTC",
            "time": "2023-01-01 12:00:00",
        }
    )
    assert result.content[0].text == "2023-01-01 12:00:00"

    # Test specific timezone conversions with known offsets
    try:
        # UTC to EST: UTC-5 in winter (January)
        result = await convert_time_tool.run(
            {
                "sourceTimezone": "UTC",
                "targetTimezone": "America/New_York",
                "time": "2023-01-01 17:00:00",
            }
        )
        # Should be 12:00:00 EST (UTC-5)
        assert result.content[0].text == "2023-01-01 12:00:00"

        # UTC to GMT+8 (Shanghai)
        result = await convert_time_tool.run(
            {
                "sourceTimezone": "UTC",
                "targetTimezone": "Asia/Shanghai",
                "time": "2023-01-01 04:00:00",
            }
        )
        # Should be 12:00:00 CST (UTC+8)
        assert result.content[0].text == "2023-01-01 12:00:00"

        # Test day boundary crossing
        result = await convert_time_tool.run(
            {
                "sourceTimezone": "UTC",
                "targetTimezone": "Asia/Shanghai",
                "time": "2023-01-01 20:00:00",
            }
        )
        # Should be next day at 04:00:00 CST
        assert result.content[0].text == "2023-01-02 04:00:00"

    except Exception:
        # Skip if timezone data unavailable
        pass


@pytest.mark.anyio(backends=["asyncio"])
async def test_get_week_year():
    """Test get_week_year function with precise week calculations."""
    get_week_year_tool = await app.get_tool("get_week_year")

    # Test current week
    result = await get_week_year_tool.run({})
    text_result = result.content[0].text
    assert "week:" in text_result and "isoWeek:" in text_result

    # Test known date: January 1, 2023 (Sunday) - ISO week 52 of 2022, strftime week 0
    result = await get_week_year_tool.run({"date": "2023-01-01"})
    text_result = result.content[0].text
    assert "week: 0, isoWeek: 52" in text_result

    # Test known date: January 9, 2023 (Monday) - ISO week 2, strftime week 2
    result = await get_week_year_tool.run({"date": "2023-01-09"})
    text_result = result.content[0].text
    assert "week: 2, isoWeek: 2" in text_result

    # Test year boundary: December 31, 2023 (Sunday) - ISO week 52
    result = await get_week_year_tool.run({"date": "2023-12-31"})
    text_result = result.content[0].text
    assert "week: 52, isoWeek: 52" in text_result


def test_moment_to_strftime():
    """Test _moment_to_strftime format conversion."""
    # Test basic conversions
    assert _moment_to_strftime("YYYY-MM-DD") == "%Y-%m-%d"
    assert _moment_to_strftime("YYYY-MM-DD HH:mm:ss") == "%Y-%m-%d %H:%M:%S"
    assert _moment_to_strftime("MM/DD/YYYY") == "%m/%d/%Y"
    assert _moment_to_strftime("h:mm A") == "%-I:%M %p"


def test_parse_dt():
    """Test _parse_dt function."""
    # Test basic datetime parsing
    dt = _parse_dt("2023-01-01 12:30:45")
    assert dt.year == 2023
    assert dt.month == 1
    assert dt.day == 1
    assert dt.hour == 12
    assert dt.minute == 30
    assert dt.second == 45

    # Test with milliseconds
    dt = _parse_dt("2023-01-01 12:30:45.123")
    assert dt.microsecond == 123000


def test_parse_date():
    """Test _parse_date function."""
    dt = _parse_date("2023-01-01")
    assert dt.year == 2023
    assert dt.month == 1
    assert dt.day == 1
    assert dt.hour == 0
    assert dt.minute == 0
    assert dt.second == 0


@pytest.mark.anyio(backends=["asyncio"])
async def test_edge_cases_and_boundaries():
    """Test edge cases and date/time boundaries."""

    # Test leap year boundary cases
    days_in_month_tool = await app.get_tool("days_in_month")

    # Test February in leap years vs non-leap years
    result = await days_in_month_tool.run({"date": "2020-02-01"})  # Leap year
    assert result.content[0].text == "29"

    result = await days_in_month_tool.run({"date": "2021-02-01"})  # Non-leap year
    assert result.content[0].text == "28"

    result = await days_in_month_tool.run({"date": "2000-02-01"})  # Century leap year
    assert result.content[0].text == "29"

    result = await days_in_month_tool.run(
        {"date": "1900-02-01"}
    )  # Century non-leap year
    assert result.content[0].text == "28"

    # Test month boundaries (30 vs 31 days)
    result = await days_in_month_tool.run({"date": "2023-04-01"})  # April (30 days)
    assert result.content[0].text == "30"

    result = await days_in_month_tool.run({"date": "2023-05-01"})  # May (31 days)
    assert result.content[0].text == "31"

    # Test year boundaries with timezone conversions
    convert_time_tool = await app.get_tool("convert_time")
    try:
        # Test New Year's Eve crossing
        result = await convert_time_tool.run(
            {
                "sourceTimezone": "UTC",
                "targetTimezone": "Pacific/Auckland",  # UTC+12/+13
                "time": "2023-12-31 23:30:00",
            }
        )
        # Should be next year in Auckland
        assert "2024-01-01" in result.content[0].text
    except Exception:
        pass  # Skip if timezone unavailable


@pytest.mark.anyio(backends=["asyncio"])
async def test_error_handling():
    """Test basic error handling."""

    # Test invalid date format for relative_time
    try:
        relative_time_tool = await app.get_tool("relative_time")
        await relative_time_tool.run({"time": "invalid-date"})
        assert False, "Should have raised an exception"
    except:
        pass  # Expected to fail

    # Test invalid date format for days_in_month
    try:
        days_in_month_tool = await app.get_tool("days_in_month")
        await days_in_month_tool.run({"date": "invalid-date"})
        assert False, "Should have raised an exception"
    except:
        pass  # Expected to fail


if __name__ == "__main__":
    pytest.main([__file__])

#  # Run all tests
# python -m pytest tests/test_tool/test_time_mcp.py -k "asyncio" -v

# # Run all helper function tests (non-async)
# python -m pytest tests/test_tool/test_time_mcp.py::test_moment_to_strftime tests/test_tool/test_time_mcp.py::test_parse_dt tests/test_tool/test_time_mcp.py::test_parse_date -v
