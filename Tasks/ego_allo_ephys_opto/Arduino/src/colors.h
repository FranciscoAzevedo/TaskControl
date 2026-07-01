class Color
{
public:
    Color();
    Color(int r, int g, int b);

    int getR() const;
    int getG() const;
    int getB() const;

private:
    int _r;
    int _g;
    int _b;
};