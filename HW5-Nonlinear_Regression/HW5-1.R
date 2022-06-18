require(ggplot2)

x <- c(1, 1, 1, 1, 
       2, 2, 2, 2, 
       3, 3, 3, 3, 
       5, 5, 5, 5, 
       7, 7, 7, 7, 
       10,10,10,10)

y <- c(105,  97, 104, 106,
       136, 161, 151, 153,
       173, 179, 174, 174,
       195, 182, 201, 172,
       207, 194, 206, 213,
       218, 193, 235, 229)

df <- data.frame(Day = x, Oxygen = y)
str(df)
summary(df)
# plot(df, main='Scatter Plot')


model_NL <- nls(y ~ theta1 * (1- exp(-theta2 * x)),
                data = df,
                start = list(theta1 = 300, theta2 = 10),
                trace = T
               )
summary(model_NL)

x_pos = 4.5
y_pos = 110
fig <- ggplot(df, aes(Day, Oxygen), main = 'nonlinear regression')
fig + geom_point(size = 2) + 
      geom_line(aes(x, fitted(model_NL)), col = 'red') + 
      annotate("point", x = x_pos, y = y_pos,
                shape = 15, color = 'red', size = 3
              ) +
      annotate("text", x = x_pos+0.2, y = y_pos,
               label = 'Nonlinear Regressor', color = 'red', hjust=0
              )
