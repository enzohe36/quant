
as_tdate <- function(date) {
  date <- as_date(date)

  holiday_list <- c(
    c(
      mdy("January 1, 2023"):mdy("January 2, 2023"),
      mdy("January 21, 2023"):mdy("January 29, 2023"),
      mdy("April 5, 2023"),
      mdy("April 29, 2023"):mdy("May 3, 2023"),
      mdy("April 23, 2023"),
      mdy("May 6, 2023"),
      mdy("June 22, 2023"):mdy("June 25, 2023"),
      mdy("September 29, 2023"):mdy("October 8, 2023"),
      mdy("December 30, 2023"):mdy("December 31, 2023")
    ),
    c(
      mdy("January 1, 2024"),
      mdy("February 9, 2024"):mdy("February 17, 2024"),
      mdy("February 4, 2024"),
      mdy("February 18, 2024"),
      mdy("April 4, 2024"):mdy("April 6, 2024"),
      mdy("April 7, 2024"),
      mdy("May 1, 2024"):mdy("May 5, 2024"),
      mdy("April 28, 2024"),
      mdy("May 11, 2024"),
      mdy("June 10, 2024"),
      mdy("September 15, 2024"):mdy("September 17, 2024"),
      mdy("September 14, 2024"),
      mdy("October 1, 2024"):mdy("October 7, 2024"),
      mdy("September 29, 2024"),
      mdy("October 12, 2024")
    )
  ) %>%
    as_date()

  trading_date <- seq(date - weeks(2), date, "1 day") %>%
    .[!wday(.) %in% c(1, 7)] %>%
    .[!. %in% holiday_list] %>%
    .[. <= date] %>%
    last()
  ifelse(
    length(holiday_list[year(holiday_list) == year(date)]) == 0,
    stop(glue("Holidays of {year(date)} not found")),
    return(trading_date)
  )
}
