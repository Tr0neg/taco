#ifndef TACO_TEMPORAL_H
#define TACO_TEMPORAL_H

#include <cstdint>
#include <stdexcept>
#include <string>

namespace taco {

/// A calendar date in the proleptic Gregorian calendar.
/// Dates are stored as days relative to 1970-01-01.
struct Date {
  int64_t daysSinceEpoch;

  Date() : daysSinceEpoch(0) {}
  explicit Date(int64_t days) : daysSinceEpoch(days) {}
  Date(int year, unsigned month, unsigned day)
      : daysSinceEpoch(daysFromCivil(year, month, day)) {}

  int64_t getDaysSinceEpoch() const { return daysSinceEpoch; }

  friend bool operator==(Date a, Date b) {
    return a.daysSinceEpoch == b.daysSinceEpoch;
  }
  friend bool operator!=(Date a, Date b) { return !(a == b); }

private:
  static int64_t daysFromCivil(int year, unsigned month, unsigned day) {
    if (month < 1 || month > 12 || day < 1 || day > daysInMonth(year, month)) {
      throw std::invalid_argument("Invalid calendar date");
    }

    year -= month <= 2;
    const int era = (year >= 0 ? year : year - 399) / 400;
    const unsigned yearOfEra = static_cast<unsigned>(year - era * 400);
    const unsigned adjustedMonth = month > 2 ? month - 3 : month + 9;
    const unsigned dayOfYear = (153 * adjustedMonth + 2) / 5 + day - 1;
    const unsigned dayOfEra = yearOfEra * 365 + yearOfEra / 4 -
                              yearOfEra / 100 + dayOfYear;
    return static_cast<int64_t>(era) * 146097 + dayOfEra - 719468;
  }

  static unsigned daysInMonth(int year, unsigned month) {
    static const unsigned monthLengths[] = {
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2) {
      const bool leapYear = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
      return leapYear ? 29 : 28;
    }
    return monthLengths[month - 1];
  }
};

/// A UTC date and time with microsecond precision.
/// The value is stored as microseconds relative to 1970-01-01 00:00:00 UTC.
struct DateTime {
  int64_t microsecondsSinceEpoch;

  DateTime() : microsecondsSinceEpoch(0) {}
  explicit DateTime(int64_t microseconds) : microsecondsSinceEpoch(microseconds) {}
  DateTime(Date date, unsigned hour, unsigned minute, unsigned second,
           unsigned microsecond = 0)
      : microsecondsSinceEpoch(toMicroseconds(date, hour, minute, second,
                                               microsecond)) {}

  int64_t getMicrosecondsSinceEpoch() const { return microsecondsSinceEpoch; }

  friend bool operator==(DateTime a, DateTime b) {
    return a.microsecondsSinceEpoch == b.microsecondsSinceEpoch;
  }
  friend bool operator!=(DateTime a, DateTime b) { return !(a == b); }

private:
  static int64_t toMicroseconds(Date date, unsigned hour, unsigned minute,
                                unsigned second, unsigned microsecond) {
    if (hour > 23 || minute > 59 || second > 59 || microsecond > 999999) {
      throw std::invalid_argument("Invalid time of day");
    }
    const int64_t microsPerSecond = 1000000;
    const int64_t microsPerDay = 86400 * microsPerSecond;
    return date.daysSinceEpoch * microsPerDay +
           static_cast<int64_t>(hour) * 3600 * microsPerSecond +
           static_cast<int64_t>(minute) * 60 * microsPerSecond +
           static_cast<int64_t>(second) * microsPerSecond + microsecond;
  }
};

}  // namespace taco

#endif
