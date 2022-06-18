require(ggplot2)

x <- rep(c(1, 2, 3, 5, 7, 10),each=4)

y <- c(105,  97, 104, 106,
       136, 161, 151, 153,
       173, 179, 174, 174,
       195, 182, 201, 172,
       207, 194, 206, 213,
       218, 193, 235, 229)

df <- data.frame(Day = x, Oxygen = y)
str(df)
summary(df)


model_NL <- nls(y ~ theta1 * (1- exp(-theta2 * x)),
                data = df,
                start = list(theta1 = 300, theta2 = 10),
                trace = T
               )
summary(model_NL)

x_pos = 8
y_pos = c(110, 120)
fig <- ggplot(df, aes(Day, Oxygen), main = 'nonlinear regression')
fig + geom_point(size = 2) + 
  geom_line(aes(x, fitted(model_NL)), col = 'red') + 
  geom_smooth(se = FALSE, method = "lm", col = 'green') + 
  annotate("point", x = x_pos, y = y_pos,
           shape = 15, color = c('green','red'), size = 3
          ) +
  annotate("text", x = x_pos+0.15, y = y_pos,
           label = c('Linearization', 'Nonlinear'),
           color = c('green','red'), hjust=0
          )

model_lm <- lm(y ~ x)
summary(model_lm)